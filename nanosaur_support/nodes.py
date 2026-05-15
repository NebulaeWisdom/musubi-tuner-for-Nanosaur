import logging
import math

import torch

from .model import NanoSaurTransformer2DModel
from .text_encoder import NanoSaurTokenizer, te as nanosaur_text_encoder
from .vae import NanoSaurVAE


def _preferred_first(options, preferred):
    ordered = []
    seen = set()

    for name in preferred:
        if name in options and name not in seen:
            ordered.append(name)
            seen.add(name)

    for name in options:
        if name not in seen:
            ordered.append(name)

    return ordered


def _count_blocks(state_dict_keys, prefix_string):
    count = 0
    while True:
        found = False
        for key in state_dict_keys:
            if key.startswith(prefix_string.format(count)):
                found = True
                break
        if not found:
            break
        count += 1
    return count


class NanoSaurLatentFormat:
    latent_channels = 96
    spacial_downscale_ratio = 16

    def __init__(self):
        self.scale_factor = 2.3623
        self.shift_factor = -0.0179

    def process_in(self, latent):
        return (latent - self.shift_factor) / self.scale_factor

    def process_out(self, latent):
        return latent * self.scale_factor + self.shift_factor


class NanoSaurModelConfig:
    unet_extra_config = {}
    sampling_settings = {
        "multiplier": 1.0,
        "shift": 4.0,
    }
    latent_format = NanoSaurLatentFormat
    supported_inference_dtypes = [torch.bfloat16, torch.float32]
    preferred_inference_dtype = torch.bfloat16
    memory_usage_factor = 0.6

    def __init__(self, unet_config):
        self.unet_config = unet_config.copy()
        self.sampling_settings = self.sampling_settings.copy()
        self.latent_format = self.latent_format()
        self.optimizations = {"fp8": False}
        for x in self.unet_extra_config:
            self.unet_config[x] = self.unet_extra_config[x]
        self.manual_cast_dtype = None
        self.custom_operations = None
        self.quant_config = None

    def model_type(self, state_dict, prefix=""):
        return "FLOW"

    def get_model(self, state_dict, prefix="", device=None):
        return NanoSaurModel(self, device=device)

    def set_inference_dtype(self, dtype, manual_cast_dtype):
        self.unet_config['dtype'] = dtype
        self.manual_cast_dtype = manual_cast_dtype


class NanoSaurModel(torch.nn.Module):
    def __init__(self, model_config, device=None):
        super().__init__()
        self.model_config = model_config
        self.latent_format = model_config.latent_format
        self.manual_cast_dtype = model_config.manual_cast_dtype
        unet_config = model_config.unet_config
        self.diffusion_model = NanoSaurTransformer2DModel(
            in_channels=unet_config.get("in_channels", 96),
            num_groups=unet_config.get("num_groups", 16),
            hidden_size=unet_config.get("hidden_size", 1536),
            decoder_hidden_size=unet_config.get("decoder_hidden_size", 2048),
            num_encoder_blocks=unet_config.get("num_encoder_blocks", 26),
            num_decoder_blocks=unet_config.get("num_decoder_blocks", 3),
            num_text_blocks=unet_config.get("num_text_blocks", 2),
            patch_size=unet_config.get("patch_size", 1),
            txt_embed_dim=unet_config.get("txt_embed_dim", 640),
            device=device,
        )
        self.diffusion_model.eval()
        self.adm_channels = 0
        self.concat_keys = ()
        logging.info("model weight dtype %s, manual cast: %s", self.get_dtype(), self.manual_cast_dtype)

    def get_dtype(self):
        return self.diffusion_model.dtype

    def get_dtype_inference(self):
        dtype = self.get_dtype()
        if self.manual_cast_dtype is not None:
            dtype = self.manual_cast_dtype
        return dtype

    def process_latent_in(self, latent):
        return self.latent_format.process_in(latent)

    def process_latent_out(self, latent):
        return self.latent_format.process_out(latent)

    def apply_model(self, x, timestep, context=None, **kwargs):
        dtype = self.get_dtype_inference()
        x = x.to(dtype)
        if context is not None:
            context = context.to(dtype)
        timestep = timestep.float()
        model_output = self.diffusion_model(x, timestep, context=context, **kwargs)
        return model_output.float()


class NanoSaurVAEWrapper:
    def __init__(self, sd=None, device=None, dtype=None, metadata=None):
        if sd is None:
            raise RuntimeError("NanoSaur VAE weights are required.")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        if dtype is None:
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.vae_dtype = dtype

        self.first_stage_model = NanoSaurVAE(latent_dim=sd["semantic_encoder.in_proj.weight"].shape[0])
        self.memory_used_encode = lambda shape, target_dtype: (200 * shape[2] * shape[3]) * torch.tensor([], dtype=target_dtype).element_size()
        self.memory_used_decode = lambda shape, target_dtype: (400 * shape[2] * shape[3] * 16 * 16) * torch.tensor([], dtype=target_dtype).element_size()
        self.downscale_ratio = 16
        self.upscale_ratio = 16
        self.latent_channels = sd["semantic_encoder.in_proj.weight"].shape[0]
        self.latent_dim = 2
        self.output_channels = 3
        self.working_dtypes = [torch.bfloat16, torch.float32]
        self.disable_offload = True
        self.first_stage_model = self.first_stage_model.eval()
        self.first_stage_model.to(self.vae_dtype)
        self.output_device = device

        missing, unexpected = self.first_stage_model.load_state_dict(sd, strict=False)
        if len(missing) > 0:
            logging.warning("Missing NanoSaur VAE keys %s", missing)
        if len(unexpected) > 0:
            logging.debug("Leftover NanoSaur VAE keys %s", unexpected)

        logging.info(
            "NanoSaur VAE load device: %s, dtype: %s",
            self.device,
            self.vae_dtype,
        )

    def model_size(self):
        return 0

    def vae_dtype_size(self):
        return torch.tensor([], dtype=self.vae_dtype).element_size()


def _infer_nanosaur_unet_config(state_dict):
    state_dict_keys = list(state_dict.keys())
    in_channels = state_dict["dec_net.final_layer.linear.weight"].shape[0]
    patch_tokens = state_dict["s_embedder.proj.weight"].shape[1]
    return {
        "image_model": "nanosaur",
        "in_channels": in_channels,
        "patch_size": round(math.sqrt(patch_tokens / in_channels)),
        "hidden_size": state_dict["s_embedder.proj.weight"].shape[0],
        "decoder_hidden_size": state_dict["dec_net.input_proj.weight"].shape[0],
        "num_encoder_blocks": _count_blocks(state_dict_keys, "blocks.{}."),
        "num_decoder_blocks": _count_blocks(state_dict_keys, "dec_net.res_blocks.{}."),
        "num_text_blocks": _count_blocks(state_dict_keys, "text_refine_blocks.{}."),
        "num_groups": state_dict["s_embedder.proj.weight"].shape[0] // state_dict["blocks.0.attn.q_norm.weight"].shape[0],
        "txt_embed_dim": state_dict["y_embedder.proj.weight"].shape[1],
    }


def load_nanosaur_model(checkpoint_path, device=None, dtype=None):
    from safetensors.torch import load_file as load_safetensors_file

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    state_dict = load_safetensors_file(checkpoint_path, device="cpu")
    
    # Remove any prefix
    diffusion_model_prefix = "model.diffusion_model."
    stripped_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(diffusion_model_prefix):
            stripped_state_dict[key[len(diffusion_model_prefix):]] = value
        else:
            stripped_state_dict[key] = value
    if len(stripped_state_dict) > 0:
        state_dict = stripped_state_dict

    parameters = sum(v.nelement() for v in state_dict.values())
    weight_dtype = max(state_dict.values(), key=lambda v: v.nelement()).dtype

    model_config = NanoSaurModelConfig(_infer_nanosaur_unet_config(state_dict))

    if dtype is None:
        preferred_dtype = getattr(model_config, "preferred_inference_dtype", None)
        if preferred_dtype in model_config.supported_inference_dtypes:
            unet_dtype = preferred_dtype
        else:
            unet_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    else:
        unet_dtype = dtype

    model_config.set_inference_dtype(unet_dtype, None)

    model = model_config.get_model(state_dict, "")
    model.to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model


def load_nanosaur_text_encoder(checkpoint_path, device=None, dtype=None):
    from safetensors.torch import load_file as load_safetensors_file

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    clip_state_dict = load_safetensors_file(checkpoint_path, device="cpu")

    if "lm_head.weight" in clip_state_dict:
        clip_state_dict["model.lm_head.weight"] = clip_state_dict.pop("lm_head.weight")

    class ClipTarget:
        params = {}
        clip = nanosaur_text_encoder(
            **comfy.text_encoders.hunyuan_video.llama_detect(clip_state_dict)
        )
        tokenizer = NanoSaurTokenizer

    # Simplified loading
    from .text_encoder import Gemma3_270M, Gemma3_270M_Config
    config_dict = {
        "vocab_size": 262144,
        "hidden_size": 640,
        "intermediate_size": 2048,
        "num_hidden_layers": 18,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "max_position_embeddings": 32768,
        "rms_norm_eps": 1e-6,
        "head_dim": 256,
        "rope_theta": [1000000.0, 10000.0],
        "transformer_type": "gemma3",
        "qkv_bias": False,
        "sliding_window": 512,
    }
    text_encoder = Gemma3_270M(config_dict, dtype=dtype or torch.bfloat16, device=device)
    text_encoder.load_state_dict(clip_state_dict, strict=False)
    text_encoder.eval()

    from .text_encoder import NanoSaurTokenizer
    tokenizer = NanoSaurTokenizer(tokenizer_data={"spiece_model": clip_state_dict.get("spiece_model", None)})

    return tokenizer, text_encoder


def load_nanosaur_vae(checkpoint_path, device=None, dtype=None):
    from safetensors.torch import load_file as load_safetensors_file

    vae_state_dict = load_safetensors_file(checkpoint_path, device="cpu")
    return NanoSaurVAEWrapper(sd=vae_state_dict, device=device, dtype=dtype)


class NanoSaurLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (["nanosaur_diffusion_model.safetensors"],),
                "text_encoder_name": (["nanosaur_text_encoder.safetensors"],),
                "vae_name": (["nanosaur_vae_decoder.safetensors"],),
                "uncond_crossover_percent": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "NanoSaur sampler patch",
                    },
                ),
            },
            "optional": {
                "weight_dtype": (["default"], {"advanced": True}),
                "clip_device": (["default", "cpu"], {"advanced": True}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_nanosaur"
    CATEGORY = "loaders"
    DESCRIPTION = "Loads NanoSaur from a custom-node package."

    def load_nanosaur(
        self,
        unet_name,
        text_encoder_name,
        vae_name,
        uncond_crossover_percent,
        weight_dtype="default",
        clip_device="default",
    ):
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        model = load_nanosaur_model(unet_name, model_options=model_options)
        clip = load_nanosaur_text_encoder(text_encoder_name, clip_device=clip_device)
        vae = load_nanosaur_vae(vae_name)
        return (model, clip, vae)


NODE_CLASS_MAPPINGS = {
    "NanoSaurLoader": NanoSaurLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoSaurLoader": "Load NanoSaur",
}