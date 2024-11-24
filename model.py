import torch

import math

from typing import Optional
from dataclasses import dataclass

    
@dataclass
class transformer_attention_config:
    n_heads: int

    key_size: int
    value_size: int

    dropout_rate: float


@dataclass
class transformer_block_config:
    attention_config: transformer_attention_config

    dropout_rate: float


@dataclass
class transformer_network_config:
    vocab_size: int
    embedding_size: int

    dropout_rate: float

    max_sequence_length: int
    
    block_configs: list[transformer_block_config]



class xpos(torch.nn.Module):
    def __init__(self, key_head_size: int, max_sequence_length: int = 1024):
        super(xpos, self).__init__()

        if key_head_size % 2 != 0:
            raise ValueError("key head size must be divisible by 2 for the positional embedding")

        theta_base = 10000
        alpha = 0.4 * key_head_size

        drange = torch.arange(start = 2, end = key_head_size + 2, step = 2, dtype = torch.float32)
        theta = torch.pow(1 / theta_base, drange / key_head_size).repeat_interleave(2)
        zeta = ((drange / (key_head_size / 2) + alpha) / (1 + alpha)).repeat_interleave(2)
        # no effect except for numerical stability
        scale_base = 512
        # no effect except for numerical stability
        half_max_sequence_length = max_sequence_length // 2

        seq_range = torch.arange(- half_max_sequence_length, max_sequence_length - half_max_sequence_length, dtype = torch.float32).view(-1, 1, 1) / scale_base

        self.c = torch.nn.Buffer(torch.cos(seq_range * theta.view(1, 1, -1)))
        self.s = torch.nn.Buffer(torch.sin(seq_range * theta.view(1, 1, -1)))
        self.t = torch.nn.Buffer((zeta.view(1, 1, -1) ** seq_range))
        self.invt = torch.nn.Buffer(1 / self.t)



    def rotate_every_two(self, input: torch.Tensor) -> torch.Tensor:
        return torch.stack((-input[..., 1::2], input[..., 0::2]), dim = -1).flatten(-2)


    def forward(self, queries, keys, start, end) -> tuple[torch.Tensor, torch.Tensor]:
        queries = (queries * self.c[start:end] + self.rotate_every_two(queries) * self.s[start:end]) * self.t[start:end]
        keys = (keys * self.c[start:end] + self.rotate_every_two(keys) * self.s[start:end]) * self.invt[start:end]


        return queries, keys

class transformer_cache(torch.nn.Module):
    def __init__(self, network_config, max_sequence_length: int = 1024, proceeding_dimensions: Optional[list[int]] = None, device: Optional[str] = None):
        super(transformer_cache, self)

        self.max_sequence_length = max_sequence_length

        self.last_position = 0
        self.current_position = 0

        if proceeding_dimensions is None:
            proceeding_dimensions = [1]

        self.device_kwarg = {}
        if not (device is None):
            self.device_kwarg["device"] = device

        self.keys = [(
            torch.empty(
                *proceeding_dimensions,
                max_sequence_length,
                block_config.attention_config.n_attn_heads,
                block_config.attention_config.key_size // block_config.attention_config.n_attn_heads,
                **self.device_kwarg
            ),
        ) for block_config in network_config.block_configs]

        self.values = [(
            torch.empty(
                *proceeding_dimensions,
                max_sequence_length,
                block_config.n_attn_heads,
                block_config.value_size // block_config.n_attn_heads,
                **self.device_kwarg
            )
        ) for block_config in network_config.block_configs]

    def increment_position(self, amount: int):
        self.last_position = self.current_position
        self.current_position += amount

        if self.current_position > self.max_sequence_length:
            raise ValueError("total sequence length is larger than the maximum sequence length of the cache")

    def get_past_keys(self, block_number):
        if self.last_position == 0:
            raise ValueError("Cannot get past keys while last position is zero")

        return self.keys[block_number][..., :self.last_position, :, :]
    
    def get_full_keys(self, block_number):
        return self.keys[block_number][..., :self.current_position, :, :]
    
    def get_past_values(self, block_number):
        if self.last_position == 0:
            raise ValueError("Cannot get past values while last position is zero")

        return self.values[block_number][..., :self.last_position, :, :]
    
    def get_full_values(self, block_number):
        return self.values[block_number][..., :self.current_position, :, :]

    def append_keys(self, block_number, keys):
        if keys.dim() <= 2:
            keys.unsqueeze(-3)
        
        self.keys[block_number][..., self.last_position:self.current_position] = keys

    def append_values(self, block_number, values):
        if values.dim() <= 2:
            values.unsqueeze(-3)
        
        self.values[block_number][..., self.last_position:self.current_position] = values

    def get_mask(self):
        return torch.ones(
            (self.current_position - self.last_position + 1, self.current_position + 1), dtype=torch.bool, **self.device_kwarg
        ).tril(self.last_position)



class attention(torch.nn.Module):
    def __init__(self, config, block_config, network_config):
        super(attention, self).__init__()

        self.n_heads = config.n_heads
        if config.key_size % config.n_heads != 0:
            raise ValueError("key size must be divisible by the number of attention heads")
        self.key_head_size = config.key_size // config.n_heads

        if config.value_size % config.n_heads != 0:
            raise ValueError("value size must be divisible by the number of attention heads")
        self.value_head_size = config.value_size // config.n_heads


        self.query_layer = torch.nn.Linear(network_config.embedding_size, config.key_size, bias = False)
        self.key_layer = torch.nn.Linear(network_config.embedding_size, config.key_size, bias = False)
        self.value_layer = torch.nn.Linear(network_config.embedding_size, config.value_size, bias = False)

        torch.nn.init.normal_(self.query_layer.weight, mean = 0, std = 0.02)
        torch.nn.init.normal_(self.key_layer.weight, mean = 0, std = 0.02)
        torch.nn.init.normal_(self.value_layer.weight, mean = 0, std = 0.02)

        self.attention_down = torch.nn.Linear(config.value_size, network_config.embedding_size, bias = False)
        torch.nn.init.normal_(self.attention_down.weight, mean = 0, std = 0.02 / math.sqrt(len(network_config.block_configs)))


        self.dropout_rate = config.dropout_rate

        self.position_embedding = xpos(self.key_head_size, max_sequence_length = network_config.max_sequence_length)
            

    def forward(self, activations: torch.Tensor, cache: Optional[transformer_cache]) -> torch.Tensor:
        use_cache = not (cache is None)

        queries = self.query_layer(activations).unflatten(-1, (self.n_heads, self.key_head_size))

        keys = self.key_layer(activations).unflatten(-1, (self.n_heads, self.key_head_size))
        values = self.value_layer(activations).unflatten(-1, (self.n_heads, self.value_head_size))


        queries, keys = self.position_embedding(queries, keys, cache.last_position, cache.current_position)
        

        attn_mask_kwarg = {}
        if not (use_cache):
            cache.append_keys(self.block_number, keys)
            cache.append_values(self.block_number, values)
            attn_mask_kwarg['attn_mask'] = cache.get_mask()

        # transpose to switch the sequence and head dimensions
        attention = torch.nn.functional.scaled_dot_product_attention(
            queries.transpose(-3, -2),
            keys.transpose(-3, -2) if use_cache else cache.get_full_keys(self.block_number).transpose(-3, -2),
            values.transpose(-3, -2) if use_cache else cache.get_full_values(self.block_number).transpose(-3, -2),
            dropout_p = self.dropout_rate if self.training else 0.0,
            is_causal = not use_cache,
        ).transpose(-3, -2)


        return torch.nn.functional.dropout(
            self.attention_down(
                attention.flatten(-2)
            ),
            p = self.dropout_rate,
            training = self.training
        )



class transformer_block(torch.nn.Module):
    def __init__(self, config: transformer_block_config, network_config: transformer_network_config):
        super(transformer_block, self).__init__()

        self.first_ln = torch.nn.LayerNorm(network_config.embedding_size, bias = False)
        self.second_ln = torch.nn.LayerNorm(network_config.embedding_size, bias = False)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(network_config.embedding_size, network_config.embedding_size * 4, bias = False),
            torch.nn.GELU(),
            torch.nn.Linear(network_config.embedding_size * 4, network_config.embedding_size, bias = False)
        )

        torch.nn.init.normal_(self.mlp[0].weight, mean = 0, std = 0.02)
        torch.nn.init.normal_(self.mlp[2].weight, mean = 0, std = 0.02 / math.sqrt(len(network_config.block_configs)))

        self.attention = attention(config.attention_config, config, network_config)

    
    def forward(self, activations: torch.Tensor, cache: Optional[transformer_cache]) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        activations = activations + self.attention(self.first_ln(activations), cache)
        activations = activations + self.mlp(self.second_ln(activations))

        return activations




    
class transformer_network(torch.nn.Module):
    def __init__(self, config: transformer_network_config):
        super(transformer_network, self).__init__()

        self.config = config
        
        
        self.wte = torch.nn.Embedding(config.vocab_size, config.embedding_size)
        torch.nn.init.normal_(self.wte.weight, mean = 0, std = 0.02)


        self.blocks = torch.nn.ModuleList([transformer_block(block_config, config) for block_config in config.block_configs])

        for i, block in enumerate(self.blocks):
            block.attention.block_number = i


        self.final_ln = torch.nn.LayerNorm(config.embedding_size, bias = False)
        
        self.transformer_head = torch.nn.Linear(config.embedding_size, config.vocab_size, bias = False)
        self.transformer_head.weight = self.wte.weight


    def forward(self, encodings: torch.Tensor, cache: Optional[transformer_cache]) -> torch.Tensor:
        embeddings = torch.nn.functional.dropout(self.wte(encodings), p = self.config.dropout_rate, training = self.training)
        if cache is not None:
            cache.increment_position(embeddings.size(dim = -2))

        for block in self.blocks:
            embeddings = block(embeddings, cache)
        
        embeddings = self.final_ln(embeddings)
        logits = self.transformer_head(embeddings)
        return logits
