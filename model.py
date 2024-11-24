import torch

import math

from typing import Optional
from dataclasses import dataclass

    
@dataclass
class transformer_config:
    vocab_size: int
    
    
    embedding_size: int

    dropout_rate: float


    n_attn_heads: int

    key_size: int
    value_size: int


    max_sequence_length: int
    

    n_blocks: int



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
    def __init__(self, config: transformer_config, proceeding_dimensions: tuple[int, ...], device: Optional[str] = None):
        super(transformer_cache, self).__init__()

        self.config = config


        self.last_position = 0
        self.current_position = 0


        device_kwarg = {} if device is None else {'device': device}

        self.keys = torch.empty(proceeding_dimensions + (config.n_blocks, config.max_sequence_length, config.n_attn_heads, config.key_size), **device_kwarg)
        self.values = torch.empty(proceeding_dimensions + (config.n_blocks, config.max_sequence_length, config.n_attn_heads, config.value_size), **device_kwarg)

    def increment_position(self, amount: int):
        self.last_position = self.current_position
        self.current_position += amount

    def reset(self):
        self.last_position = 0
        self.current_position = 0

    def append_keys(self, keys: torch.Tensor, block_number: int):
        self.keys[..., block_number, self.last_position:self.current_position, :, :] = keys

    def append_values(self, values: torch.Tensor, block_number: int):
        self.values[..., block_number, self.last_position:self.current_position, :, :] = values

    def get_full_keys(self, block_number: int) -> torch.Tensor:
        return self.keys[..., block_number, :self.current_position, :, :]
    
    def get_full_values(self, block_number: int) -> torch.Tensor:
        return self.values[..., block_number, :self.current_position, :, :]

    def get_previous_keys(self, block_number: int) -> torch.Tensor:
        return self.keys[..., block_number, :self.last_position, :, :]

    def get_previous_values(self, block_number: int) -> torch.Tensor:
        return self.values[..., block_number, :self.last_position, :, :]
    

    def get_mask(self) -> torch.Tensor:
        return torch.ones(
            (self.current_position - self.last_position + 1, self.current_position + 1), dtype=torch.bool, **self.device_kwarg
        ).tril(self.last_position)
        


class transformer_attention(torch.nn.Module):
    def __init__(self, key_size: int, value_size: int, max_sequence_length: int):
        super(transformer_attention, self).__init__()

        self.key_size = key_size
        self.value_size = value_size

        self.position_embedding = xpos(key_size, max_sequence_length = max_sequence_length)
    




class transformer_block(torch.nn.Module):
    def __init__(self, config: transformer_config, block_number: int):
        super(transformer_block, self).__init__()

        self.config = config


        self.block_number = block_number


        if config.key_size % config.n_attn_heads != 0:
            raise ValueError("key size must be divisible by the number of attention heads")
        self.key_head_size = config.key_size // config.n_attn_heads

        if config.value_size % config.n_attn_heads != 0:
            raise ValueError("value size must be divisible by the number of attention heads")
        self.value_head_size = config.value_size // config.n_attn_heads



        self.first_ln = torch.nn.LayerNorm(config.embedding_size, bias = False)
        self.second_ln = torch.nn.LayerNorm(config.embedding_size, bias = False)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(config.embedding_size, config.embedding_size * 4, bias = False),
            torch.nn.GELU(),
            torch.nn.Linear(config.embedding_size * 4, config.embedding_size, bias = False)
        )

        torch.nn.init.normal_(self.mlp[0].weight, mean = 0, std = 0.02)
        torch.nn.init.normal_(self.mlp[2].weight, mean = 0, std = 0.02 / math.sqrt(config.n_blocks))


        self.query_layer = torch.nn.Linear(config.embedding_size, config.key_size, bias = False)
        self.key_layer = torch.nn.Linear(config.embedding_size, config.key_size, bias = False)
        self.value_layer = torch.nn.Linear(config.embedding_size, config.value_size, bias = False)

        torch.nn.init.normal_(self.query_layer.weight, mean = 0, std = 0.02)
        torch.nn.init.normal_(self.key_layer.weight, mean = 0, std = 0.02)
        torch.nn.init.normal_(self.value_layer.weight, mean = 0, std = 0.02)

        self.attention_down = torch.nn.Linear(config.value_size, config.embedding_size, bias = False)
        torch.nn.init.normal_(self.attention_down.weight, mean = 0, std = 0.02 / math.sqrt(config.n_blocks))

        self.position_embedding = xpos(self.key_head_size, max_sequence_length = config.max_sequence_length)
    

    def get_full_kv(self, incoming_kv, kv_cache, index) -> tuple[tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        if kv_cache is None:
            return incoming_kv, None
        else:
            incoming_keys, incoming_values = incoming_kv
            cache_keys, cache_values = kv_cache

            incoming_sequence_length = incoming_keys.size(-3)
            cache_sequence_length = index
            total_sequence_length = incoming_sequence_length + cache_sequence_length
            max_sequence_length = cache_keys.size(-3)

            if total_sequence_length > max_sequence_length:
                raise ValueError("total sequence length is larger than the maximum sequence length of the cache")
            elif total_sequence_length == max_sequence_length and index == 0:
                return (incoming_keys, incoming_values), None
            elif total_sequence_length == max_sequence_length:
                keys = torch.cat((cache_keys[..., :cache_sequence_length, :, :], incoming_keys), dim = -3)
                values = torch.cat((cache_values[..., :cache_sequence_length, :, :], incoming_values), dim = -3)

                mask = torch.ones((incoming_sequence_length, total_sequence_length), dtype=torch.bool, device = keys.device).tril(total_sequence_length - incoming_sequence_length)

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

                # custom mask so that it can pay attention to the cached tokens
                mask = torch.ones((incoming_sequence_length, total_sequence_length), dtype=torch.bool, device = keys.device).tril(total_sequence_length - incoming_sequence_length)

                return (keys, values), mask


    
    def forward(self, activations: torch.Tensor, cache: Optional[transformer_cache]) = None -> torch.Tensor:
        activation_norms = self.first_ln(activations)

        queries = self.query_layer(activation_norms).unflatten(-1, (self.config.n_attn_heads, self.key_head_size))

        incoming_keys = self.key_layer(activation_norms).unflatten(-1, (self.config.n_attn_heads, self.key_head_size))
        incoming_values = self.value_layer(activation_norms).unflatten(-1, (self.config.n_attn_heads, self.value_head_size))


        queries, incoming_keys = self.position_embedding(queries, incoming_keys, cache.last_position, cache.current_position)


        (keys, values), mask = self.get_full_kv((incoming_keys, incoming_values), kv_cache, index)

        # transpose to switch the sequence and head dimensions
        attention = torch.nn.functional.scaled_dot_product_attention(
            queries.transpose(-3, -2),
            keys.transpose(-3, -2),
            values.transpose(-3, -2),
            is_causal = (mask is None),
            dropout_p = self.config.dropout_rate if self.training else 0.0,
            attn_mask = mask
        ).transpose(-3, -2)


        activations = activations + torch.nn.functional.dropout(
            self.attention_down(
                attention.flatten(-2)
            ), 
            p = self.config.dropout_rate,
            training = self.training
        )

        activations = activations + self.mlp(self.second_ln(activations))

        return activations




    
class transformer_network(torch.nn.Module):
    def __init__(self, config: transformer_config):
        super(transformer_network, self).__init__()

        self.config = config

        
        self.blocks = torch.nn.ModuleList([transformer_block(config, block_number) for block_number in range(config.n_blocks)])
        
        
        self.wte = torch.nn.Embedding(config.vocab_size, config.embedding_size)
        torch.nn.init.normal_(self.wte.weight, mean = 0, std = 0.02)

        self.final_ln = torch.nn.LayerNorm(config.embedding_size, bias = False)
        
        self.lm_head = torch.nn.Linear(config.embedding_size, config.vocab_size, bias = False)
        self.lm_head.weight = self.wte.weight

    # index should start at 0
    def forward(self, encodings: torch.Tensor, cache: Optional[transformer_cache] = None) -> torch.Tensor:
        embeddings = torch.nn.functional.dropout(
            self.wte(encodings),
            p = self.config.dropout_rate,
            training = self.training
        )

        cache.increment_position(embeddings.size(-2))

        for block in self.blocks:
            embeddings = block(embeddings, cache)
        
        embeddings = self.final_ln(embeddings)
        logits = self.lm_head(embeddings)

        return logits  
