import torch
import datasets
from typing import Any, Protocol, TypeVar



tokenizer_type = TypeVar('tokenizer_type')

class pipeline_protocol(Protocol[tokenizer_type]):
    def get_dataset_and_tokenizer(self, **kwargs: Any) -> tuple[datasets.DatasetDict, tokenizer_type]:
        ...

    def save_dataset(self, dataset: datasets.DatasetDict, path: str) -> None:
        ...

    def load_dataset(self, path: str) -> datasets.DatasetDict:
        ...

    def save_tokenizer(self, tokenizer: tokenizer_type, path: str) -> None:
        ...

    def load_tokenizer(self, path: str) -> tokenizer_type:
        ...

    def encode_text(self, tokenizer: tokenizer_type, text: str) -> torch.Tensor:
        ...

    def decode_ids(self, tokenizer: tokenizer_type, ids: torch.Tensor) -> str:
        ...

    def get_vocab_size(self, tokenizer: tokenizer_type) -> int:
        ...

    def should_halt_generation(self, tokenizer: tokenizer_type, last_token_id: int) -> bool:
        ...