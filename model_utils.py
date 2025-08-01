"""
Model utilities - standalone version of state.tx.models.utils
Contains only the necessary functions for StateTransitionPerturbationModel
"""

from typing import Union

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model, LlamaConfig, LlamaModel, PreTrainedModel


def build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dim: int,
    n_layers: int,
    dropout: float = 0.0,
    activation: nn.Module = nn.ReLU,  # default to nn.ReLU class
) -> nn.Sequential:
    """
    Build an MLP of `n_layers` from `in_dim` to `out_dim`.
    """
    layers = []
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")

    if n_layers == 1:
        layers.append(nn.Linear(in_dim, out_dim))
    else:
        # First layer
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(activation())  # instantiate the class
        layers.append(nn.Dropout(dropout))

        # Intermediate layers
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())  # instantiate again
            layers.append(nn.Dropout(dropout))

        # Final layer
        layers.append(nn.Linear(hidden_dim, out_dim))

    return nn.Sequential(*layers)


def get_activation_class(name: str) -> nn.Module:
    """
    Given a string activation name, return the corresponding nn.Module class.

    Supported activation functions:
    - ReLU
    - LeakyReLU
    - ELU
    - SELU
    - GELU
    """
    name = name.lower()

    if name == "relu":
        return nn.ReLU
    elif name == "leakyrelu":
        return nn.LeakyReLU
    elif name == "elu":
        return nn.ELU
    elif name == "selu":
        return nn.SELU
    elif name == "gelu":
        return nn.GELU
    else:
        raise ValueError(f"Unsupported activation function: {name}")


class NoRoPE(nn.Module):
    """
    A drop-in replacement for LlamaRotaryEmbedding that always returns:
      cos = all ones, sin = all zeros
    of shape (batch_size, seq_len, head_dim), so rotary has no effect.
    """

    def __init__(self, num_attention_heads: int, hidden_size: int):
        super().__init__()
        self.num_heads = num_attention_heads
        self.hidden_size = hidden_size

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.LongTensor):
        # hidden_states: (batch_size, seq_len, hidden_dim)
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Create cos = ones, sin = zeros
        #   shape --> (batch_size, seq_len, head_dim)
        cos = hidden_states.new_ones(batch_size, seq_len, self.num_heads)
        sin = hidden_states.new_zeros(batch_size, seq_len, self.num_heads)
        return cos, sin


class LlamaBidirectionalModel(LlamaModel):
    """
    A drop-in replacement for LlamaModel with bidirectional attention.
    By overriding _update_causal_mask to return None, all tokens attend to each other.
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)

        self.rotary_emb = NoRoPE(
            num_attention_heads=config.head_dim,
            hidden_size=config.hidden_size,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values,
        output_attentions: bool = False,
    ):
        # By returning None, we disable any causal‐(look‐ahead) masking.
        # The only mask that remains is whatever "attention_mask" the user has passed
        # (e.g. padding‐mask), which will be handled by Flash/SDPA internally as non‐causal.
        return None

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        cache_position: torch.LongTensor = None,
        **flash_attn_kwargs,
    ):
        flash_attn_kwargs["is_causal"] = False

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **flash_attn_kwargs,
        )


class GPT2BidirectionalModel(GPT2Model):
    """
    A thin wrapper around GPT2Model that disables the causal (unidirectional) mask,
    allowing full bidirectional attention.
    """

    def __init__(self, config: GPT2Config):
        # Mark as not‐a‐decoder (for downstream utilities).
        config.is_decoder = False
        super().__init__(config)

        # Overwrite each attention's bias so no triangular masking occurs.
        for block in self.h:
            # block.attn.bias is a bool‐tensor of shape (1, 1, max_pos, max_pos).
            block.attn.bias.data.fill_(True)
            block.attn.is_causal = False

        def _no_causal_mask(
            self,
            attention_mask: torch.Tensor,
            input_tensor: torch.Tensor,
            cache_position: torch.Tensor,
            past_key_values,
            output_attentions: bool,
        ):
            return None

        self._update_causal_mask = _no_causal_mask.__get__(self, GPT2Model)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        cache_position=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        # Call the parent forward method
        return super().forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )


def get_transformer_backbone(key: str, kwargs: dict) -> tuple[PreTrainedModel, int]:
    """
    Get transformer backbone model and its hidden dimension.

    Args:
        key: Transformer type ("GPT2" or "llama")
        kwargs: Configuration parameters for the transformer

    Returns:
        Tuple of (model, model_dim)
    """
    if key == "GPT2":
        config = GPT2Config(**kwargs)
        model = GPT2BidirectionalModel(config)

        # Zero out position embeddings and freeze them
        model.wpe.weight.requires_grad = False
        model.wte.weight.requires_grad = False
        model.wpe.weight.zero_()
        model.wte.weight.zero_()

        model_dim = config.n_embd
    elif key == "llama":
        config = LlamaConfig(**kwargs)
        model = LlamaBidirectionalModel(config)
        model_dim = config.hidden_size

        model.embed_tokens.weight.requires_grad = False
        model.embed_tokens.weight.zero_()
    else:
        raise ValueError(f"Unknown backbone key {key}")

    return model, model_dim
