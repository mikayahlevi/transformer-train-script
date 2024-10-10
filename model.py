import torch

import math

from typing import Optional
from dataclasses import dataclass

    


@dataclass
class transformer_block_config:
    n_attn_heads: int

    hidden_size: int
    key_size: int
    value_size: int

    # key_head_size: int
    # value_head_size: int



@dataclass
class transformer_network_config:
    vocab_size: int
    embedding_size: int

    dropout_rate: float
    
    block_configs: list[transformer_block_config]



class xpos(torch.nn.Module):
    def __init__(self, key_head_size: int, device = 'cuda', max_sequence_length: int = 1024):
        super(xpos, self).__init__()

        self.theta_base = 10000
        self.alpha = 0.4 * key_head_size

        drange = torch.arange(start = 2, end = key_head_size + 2, step = 2, dtype = torch.float32, device = device)
        self.theta = torch.pow(1 / self.theta_base, drange / key_head_size).repeat_interleave(2)
        self.zeta = ((drange / (key_head_size / 2) + self.alpha) / (1 + self.alpha)).repeat_interleave(2)
        # no effect except for numerical stability
        scale_base = 512
        # no effect except for numerical stability
        half_max_sequence_length = max_sequence_length // 2

        self.seq_range = torch.arange(- half_max_sequence_length, max_sequence_length - half_max_sequence_length, dtype = torch.float32, device = device) / scale_base

    def rotate_every_two(self, input: torch.Tensor) -> torch.Tensor:
        return torch.stack((-input[..., 1::2], input[..., 0::2]), dim = -1).flatten(-2)


    def forward(self, queries, keys, start, end) -> tuple[torch.Tensor, torch.Tensor]:
        seq_range = self.seq_range[start:end].view(-1, 1, 1)

        c = torch.cos(seq_range * self.theta.view(1, 1, -1))
        s = torch.sin(seq_range * self.theta.view(1, 1, -1))
        t = (self.zeta.view(1, 1, -1) ** seq_range)

        queries = (queries * c + self.rotate_every_two(queries) * s) * t
        keys = (keys * c + self.rotate_every_two(keys) * s) * (t ** -1)


        return queries, keys


class transformer_block(torch.nn.Module):
    def __init__(self, network_config: transformer_network_config, block_config: transformer_block_config):
        super(transformer_block, self).__init__()

        self.block_config = block_config
        self.network_config = network_config


        self.first_ln = torch.nn.LayerNorm(block_config.hidden_size, bias = False)
        self.second_ln = torch.nn.LayerNorm(block_config.hidden_size, bias = False)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(block_config.hidden_size, block_config.hidden_size * 4, bias = False),
            torch.nn.GELU(),
            torch.nn.Linear(block_config.hidden_size * 4, block_config.hidden_size, bias = False)
        )

        torch.nn.init.normal_(self.mlp[0].weight, mean = 0, std = 0.02)
        torch.nn.init.normal_(self.mlp[2].weight, mean = 0, std = 0.02 / math.sqrt(len(network_config.block_configs)))


        self.query_layer = torch.nn.Linear(block_config.hidden_size, block_config.key_size, bias = False)
        self.key_layer = torch.nn.Linear(block_config.hidden_size, block_config.key_size, bias = False)
        self.value_layer = torch.nn.Linear(block_config.hidden_size, block_config.value_size, bias = False)

        torch.nn.init.normal_(self.query_layer.weight, mean = 0, std = 0.02)
        torch.nn.init.normal_(self.key_layer.weight, mean = 0, std = 0.02)
        torch.nn.init.normal_(self.value_layer.weight, mean = 0, std = 0.02)

        self.attention_linear = torch.nn.Linear(block_config.value_size, block_config.hidden_size, bias = False)
        torch.nn.init.normal_(self.attention_linear.weight, mean = 0, std = 0.02 / math.sqrt(len(network_config.block_configs)))

        self.position_embedding = xpos(self.block_config.key_size // self.block_config.n_attn_heads, device = 'cuda', max_sequence_length = 1024)
    

    def get_full_kv(self, incoming_kv, kv_cache, index) -> tuple[tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        incoming_keys, incoming_values = incoming_kv
        cache_keys, cache_values = kv_cache

        incoming_sequence_length = incoming_keys.size(-3)
        cache_sequence_length = index
        total_sequence_length = incoming_sequence_length + cache_sequence_length
        max_sequence_length = cache_keys.size(-3)

        if total_sequence_length > max_sequence_length:
            raise ValueError("total sequence length is greater than the maximum sequence length")
        elif total_sequence_length == max_sequence_length and index == 0:
            return (incoming_keys, incoming_values), None
        elif total_sequence_length == max_sequence_length:
            keys = torch.cat((cache_keys[..., :cache_sequence_length, :, :], incoming_keys), dim = -3)
            values = torch.cat((cache_values[..., :cache_sequence_length, :, :], incoming_values), dim = -3)

            mask = torch.ones(incoming_sequence_length, total_sequence_length, dtype = torch.bool, device = keys.device).triu()

            return (keys, values), mask
        elif index == 0:
            cache_keys[..., :incoming_sequence_length, :, :] = incoming_keys
            cache_values[..., :incoming_sequence_length, :, :] = incoming_values

            return (incoming_keys, incoming_values), None
        else:

            keys = torch.cat((cache_keys[..., :cache_sequence_length, :, :], incoming_keys), dim = -3)
            values = torch.cat((cache_values[..., :cache_sequence_length, :, :], incoming_values), dim = -3)

            cache_keys[..., cache_sequence_length:total_sequence_length, :, :] = incoming_keys
            cache_values[..., cache_sequence_length:total_sequence_length, :, :] = incoming_values

            # custom mask so that it can pay attention to the previously passed through tokens
            mask = torch.ones(incoming_sequence_length, total_sequence_length, dtype = torch.bool, device = keys.device).triu()

            return (keys, values), mask


    
    def forward(self, activations: torch.Tensor, kv_cache: tuple[torch.Tensor, torch.Tensor], index) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        activation_norms = self.first_ln(activations)

        queries = self.query_layer(activation_norms).unflatten(-1, (self.block_config.n_attn_heads, self.block_config.key_head_size))

        incoming_keys = self.key_layer(activation_norms).unflatten(-1, (self.block_config.n_attn_heads, self.block_config.key_head_size))
        incoming_values = self.value_layer(activation_norms).unflatten(-1, (self.block_config.n_attn_heads, self.block_config.value_head_size))


        incoming_keys, incoming_values = self.position_embedding(incoming_keys, incoming_values, index, index + queries.size(-3))


        (keys, values), mask = self.get_full_kv((incoming_keys, incoming_values), kv_cache, index)

        # transpose to switch the sequence and head dimensions
        attention = torch.nn.functional.scaled_dot_product_attention(
            queries.transpose(-3, -2),
            keys.transpose(-3, -2),
            values.transpose(-3, -2),
            is_causal = (mask is None),
            dropout_p = self.network_config.dropout_rate if self.training else 0.0,
            attn_mask = mask
        ).transpose(-3, -2)


        activations = activations + torch.nn.functional.dropout(
            self.attention_linear(
                attention.flatten(-2)
            ), 
            p = self.network_config.dropout_rate,
            training = self.training
        )

        activations = activations + self.mlp(self.second_ln(activations))

        return activations, kv_cache




    
class transformer_network(torch.nn.Module):
    def __init__(self, config: transformer_network_config):
        super(transformer_network, self).__init__()

        self.config = config

        for block_config in config.block_configs:
            assert block_config.key_size % block_config.n_attn_heads == 0
            block_config.key_head_size = block_config.hidden_size // block_config.n_attn_heads

            assert block_config.value_size % block_config.n_attn_heads == 0
            block_config.value_head_size = block_config.hidden_size // block_config.n_attn_heads

            self.blocks = torch.nn.ModuleList([transformer_block(config, block_config) for block_config in config.block_configs])

        
        self.wte = torch.nn.Embedding(config.vocab_size, config.embedding_size)
        torch.nn.init.normal_(self.wte.weight, mean = 0, std = 0.02)

        self.final_ln = torch.nn.LayerNorm(config.embedding_size, bias = False)
        
        self.lm_head = torch.nn.Linear(config.embedding_size, config.vocab_size, bias = False)
        self.lm_head.weight = self.wte.weight
        # torch.nn.init.normal_(self.lm_head.weight, mean = 0, std = 0.02)

    # index should start at 0
    def forward(self, encodings: torch.Tensor, kv_cache: list[tuple[torch.Tensor, torch.Tensor]], index: int) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:

        embeddings = torch.nn.functional.dropout(self.wte(encodings), p = self.config.dropout_rate, training = self.training)

        for i, block in enumerate(self.blocks):
            embeddings, kv_cache[i] = block.forward(embeddings, kv_cache[i], index)
        
        embeddings = self.final_ln(embeddings)

        logits = self.lm_head(embeddings)

        return logits, kv_cache  
        
    
    def get_empty_kv_cache(self, batch_size: int, sequence_length: int, device) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return ([
            (
                torch.empty(batch_size, sequence_length, block_config.n_attn_heads, block_config.key_head_size, device = device).squeeze(-4), # type: ignore
                torch.empty(batch_size, sequence_length, block_config.n_attn_heads, block_config.value_head_size, device = device).squeeze(-4) # type: ignore
            )
        for block_config in self.config.block_configs])