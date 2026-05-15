from __future__ import annotations

import re
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


TEXT_MAX_LENGTH = 128


@dataclass
class Gemma3_270M_Config:
    vocab_size: int = 262144
    hidden_size: int = 640
    intermediate_size: int = 2048
    num_hidden_layers: int = 18
    num_attention_heads: int = 4
    num_key_value_heads: int = 1
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta = [1000000.0, 10000.0]
    transformer_type: str = "gemma3"
    head_dim = 256
    rms_norm_add = True
    mlp_activation = "gelu_pytorch_tanh"
    qkv_bias = False
    rope_dims = None
    q_norm = "gemma3"
    k_norm = "gemma3"
    sliding_attention = [512, 512, 512, 512, 512, False]
    rope_scale = None
    final_norm: bool = True
    lm_head: bool = False
    stop_tokens = [1, 106]


class GemmaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)


class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=32768, theta=10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=x.dtype)
        sin = emb.sin().to(dtype=x.dtype)
        return cos, sin


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GemmaAttention(nn.Module):
    def __init__(self, config: Gemma3_270M_Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.qkv_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps) if config.q_norm == "gemma3" else nn.Identity()
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps) if config.k_norm == "gemma3" else nn.Identity()
        self.rotary_emb = GemmaRotaryEmbedding(self.head_dim, max_position_embeddings=config.max_position_embeddings)

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        return self.o_proj(attn_output)


class GemmaMLP(nn.Module):
    def __init__(self, config: Gemma3_270M_Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: Gemma3_270M_Config):
        super().__init__()
        self.self_attn = GemmaAttention(config)
        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask, position_ids=position_ids)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class GemmaModel(nn.Module):
    def __init__(self, config: Gemma3_270M_Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([GemmaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.embed_tokens(input_ids)
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device).unsqueeze(0)
        all_hidden_states = [hidden_states]
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask, position_ids=position_ids)
            all_hidden_states.append(hidden_states)
        hidden_states = self.norm(hidden_states)
        all_hidden_states[-1] = hidden_states
        return type('ModelOutput', (), {'hidden_states': all_hidden_states})()


class Gemma3_270M(nn.Module):
    def __init__(self, config_dict, dtype, device, operations=None):
        super().__init__()
        config = Gemma3_270M_Config(**{k: v for k, v in config_dict.items() if hasattr(Gemma3_270M_Config, k)})
        self.model = GemmaModel(config)
        self.num_layers = config.num_hidden_layers
        self.dtype = dtype

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, **kwargs):
        result = self.model(input_ids, attention_mask=attention_mask)
        if output_hidden_states:
            return type('ModelOutput', (), {'hidden_states': result.hidden_states})()
        return type('ModelOutput', (), {'last_hidden_state': result.hidden_states[-1]})()

    def load_state_dict(self, state_dict, strict=False):
        # Map HuggingFace keys to our model keys
        mapped = {}
        for key, value in state_dict.items():
            if key == "model.lm_head.weight":
                continue
            new_key = key.replace("model.layers.", "model.layers.").replace("language_model.model.", "model.")
            mapped[new_key] = value
        return super().load_state_dict(mapped, strict=False)


def parse_prompt_emphasis(caption: str) -> tuple[str, list[tuple[int, int, float]]]:
    if not caption:
        return "", []

    weight_pattern = re.compile(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)$")
    spans: list[tuple[int, int, float]] = []
    parts: list[str] = []
    cursor = 0
    output_len = 0
    idx = 0
    while idx < len(caption):
        if caption[idx] != "(":
            idx += 1
            continue

        depth = 1
        end = idx + 1
        while end < len(caption) and depth > 0:
            if caption[end] == "(":
                depth += 1
            elif caption[end] == ")":
                depth -= 1
            end += 1

        if depth != 0:
            idx += 1
            continue

        inner = caption[idx + 1:end - 1]
        inner_depth = 0
        colon_idx = -1
        for inner_idx, char in enumerate(inner):
            if char == "(":
                inner_depth += 1
            elif char == ")":
                inner_depth -= 1
            elif char == ":" and inner_depth == 0:
                colon_idx = inner_idx

        if colon_idx == -1:
            idx += 1
            continue

        emphasized_text = inner[:colon_idx]
        weight_text = inner[colon_idx + 1:].strip()
        if not emphasized_text or not weight_pattern.fullmatch(weight_text):
            idx += 1
            continue

        literal = caption[cursor:idx]
        if literal:
            parts.append(literal)
            output_len += len(literal)

        span_start = output_len
        parts.append(emphasized_text)
        output_len += len(emphasized_text)
        spans.append((span_start, output_len, float(weight_text)))
        cursor = end
        idx = end

    tail = caption[cursor:]
    if tail:
        parts.append(tail)

    return "".join(parts), spans


def token_weight_for_span(token_begin: int, token_end: int, emphasis_spans: list[tuple[int, int, float]]) -> float:
    token_weight = 1.0
    for span_begin, span_end, span_weight in emphasis_spans:
        if token_begin < span_end and token_end > span_begin:
            token_weight *= span_weight
    return token_weight


class NanosaurGemma270MTokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        import sentencepiece as spm
        import tempfile
        self.max_length = TEXT_MAX_LENGTH
        spiece_model = tokenizer_data.get("spiece_model", None)
        if spiece_model is not None:
            model_bytes = bytes(spiece_model.cpu().numpy().tolist())
            self.processor = spm.SentencePieceProcessor()
            with tempfile.NamedTemporaryFile(suffix=".model") as handle:
                handle.write(model_bytes)
                handle.flush()
                self.processor.Load(handle.name)
        else:
            self.processor = None
        self.bos_token_id = 2
        self.pad_token_id = 0
        self.start_token = self.bos_token_id

    def tokenize_with_weights(self, text: str, return_word_ids=False, **kwargs):
        cleaned_text, emphasis_spans = parse_prompt_emphasis(text)
        
        batch = []
        if self.processor is not None:
            proto_pieces = self.processor.EncodeAsIds(cleaned_text)
            for i, token_id in enumerate(proto_pieces):
                token_weight = token_weight_for_span(i, i + 1, emphasis_spans)
                batch.append((token_id, token_weight, i + 1))
        else:
            # Fallback: character-level tokenization
            for i, char in enumerate(cleaned_text):
                token_weight = token_weight_for_span(i, i + 1, emphasis_spans)
                batch.append((ord(char) % 262144, token_weight, i + 1))

        if self.start_token is not None:
            batch.insert(0, (self.start_token, 1.0, 0))

        if len(batch) > self.max_length:
            batch = batch[:self.max_length]
        
        while len(batch) < self.max_length:
            batch.append((self.pad_token_id, 1.0, 0))

        if not return_word_ids:
            batch = [(token, weight) for token, weight, _ in batch]
        return [batch]

    def state_dict(self):
        return {"spiece_model": None}


class NanoSaurTokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        self.tokenizer = NanosaurGemma270MTokenizer(
            embedding_directory=embedding_directory,
            tokenizer_data=tokenizer_data,
        )

    def tokenize_with_weights(self, text: str, **kwargs):
        return self.tokenizer.tokenize_with_weights(text, **kwargs)

    def state_dict(self):
        return self.tokenizer.state_dict()


class NanosaurGemma270MModel:
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, attention_mask=True, model_options={}):
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
        self.model = Gemma3_270M(config_dict, dtype=dtype or torch.bfloat16, device=device)
        self.device = device
        self.dtype = dtype

    def encode_token_weights(self, token_weight_pairs):
        if isinstance(token_weight_pairs, dict):
            token_weight_pairs = token_weight_pairs.get("nanosaur_gemma270m", token_weight_pairs)
        
        tokens = [[token for token, _ in section] for section in token_weight_pairs]
        input_ids = torch.tensor(tokens, device=self.device, dtype=torch.long)
        attention_mask = (input_ids != 0).to(torch.long)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        out = outputs.hidden_states[-1].to(device=self.device)
        pooled = None
        
        token_weights = torch.tensor(
            [[weight for _, weight in section] for section in token_weight_pairs],
            device=self.device,
            dtype=torch.float32,
        ).flatten().unsqueeze(dim=0)
        
        extra = {
            "token_weights": token_weights,
        }
        
        return (out, pooled, extra)


class NanoSaurClipModel:
    def __init__(self, device="cpu", dtype=None, model_options={}):
        self.clip_model = NanosaurGemma270MModel(
            device=device,
            dtype=dtype,
            model_options=model_options,
        )

    def encode_token_weights(self, token_weight_pairs):
        return self.clip_model.encode_token_weights(token_weight_pairs)


def te(dtype_llama=None, llama_quantization_metadata=None):
    class NanoSaurTEModel_(NanoSaurClipModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if dtype_llama is not None:
                dtype = dtype_llama
            super().__init__(device=device, dtype=dtype, model_options=model_options)

    return NanoSaurTEModel_