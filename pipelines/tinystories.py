from typing import Any, Optional, cast
import torch

import datasets
import transformers

from pipeline import pipeline_protocol


class main_pipeline(pipeline_protocol[datasets.DatasetDict, transformers.PreTrainedTokenizerFast]):
    def get_dataset_and_tokenizer(self, sequence_length: int) -> tuple[datasets.DatasetDict, transformers.PreTrainedTokenizerFast]:
        tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
        tokenizer.pad_token = tokenizer.eos_token


        def repack_ids(examples):
            concatenated_ids = sum(examples["input_ids"], [])
            total_length = len(concatenated_ids)

            n_chunks = total_length // sequence_length

            # we need one more token than the number of tokens in the chunks
            # so decrement n_chunks if the single extra token is not available
            if sequence_length * n_chunks + 1 > total_length:
                n_chunks -= 1

            # the last token of the ith chunk is the first token of the (i + 1)th chunk
            return {
                "input_ids": [concatenated_ids[i : i + sequence_length + 1] for i in range(0, n_chunks * sequence_length, sequence_length)]
            }


        dataset = datasets.load_dataset("roneneldan/TinyStories")

        dataset = dataset.map(
            lambda examples: tokenizer(examples["text"], truncation = False),
            batched = True,
            remove_columns = ["text"],
            num_proc = 8,
        )

        dataset = dataset.remove_columns([col for col in dataset["train"].column_names if col != "input_ids"])

        dataset = dataset.map(
            repack_ids,
            batched = True,
            num_proc = 8,
        )

        return dataset.rename_column("input_ids", "ids").with_format(type = "torch", columns = ["ids"]), tokenizer

    def get_dataloaders(self, dataset: datasets.DatasetDict, **dataloader_args: Any) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        train_dataloader = torch.utils.data.DataLoader(cast(torch.utils.data.Dataset, dataset["train"]), **dataloader_args)
        val_dataloader = torch.utils.data.DataLoader(cast(torch.utils.data.Dataset, dataset["validation"]), **dataloader_args)

        return train_dataloader, val_dataloader

    def get_training_pairs(self, batch: torch.Tensor, tokenizer: transformers.PreTrainedTokenizerFast, mask_value: int) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        return inputs, targets

    def save_dataset(self, dataset: datasets.DatasetDict, path: str) -> None:
        dataset.save_to_disk(path)

    def load_dataset(self, path: str) -> datasets.DatasetDict:
        return cast(datasets.DatasetDict, datasets.load_from_disk(path))

    def save_tokenizer(self, tokenizer: transformers.PreTrainedTokenizerFast, path: str) -> None:
        tokenizer.save_pretrained(path)

    def load_tokenizer(self, path: str) -> transformers.PreTrainedTokenizerFast:
        tokenizer = transformers.AutoTokenizer.from_pretrained(path)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def encode_text(self, tokenizer: transformers.PreTrainedTokenizerFast, text: str) -> torch.Tensor:
        return torch.tensor(tokenizer.encode(text)).long()

    def decode_ids(self, tokenizer: transformers.PreTrainedTokenizerFast, ids: torch.Tensor) -> str:
        assert ids.dim() == 1, 'ids must be a 1-dimensional tensor'
        return tokenizer.decode(ids.tolist(), skip_special_tokens = True)

    def get_vocab_size(self, tokenizer: transformers.PreTrainedTokenizerFast) -> int:
        return len(tokenizer)

    def should_halt_generation(self, tokenizer: transformers.PreTrainedTokenizerFast, last_token_id: int) -> bool:
        return last_token_id == tokenizer.eos_token_id