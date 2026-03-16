import requests
import datasets

import os

class character_tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.vocab_size = len(vocab)

        self.token_to_id_dict = {token: i for i, token in enumerate(vocab)}
        self.id_to_token_dict = {i: token for i, token in enumerate(vocab)}

        self.padding = False

        self.character_level = True

    def encode(self, text):
        return [self.token_to_id_dict[token] for token in text]

    def token_to_id(self, token):
        return self.token_to_id_dict[token]

    def decode(self, ids):
        return ''.join([self.id_to_token_dict[i] for i in ids])

    def get_vocab_size(self):
        return self.vocab_size


class main_pipeline:
    def get_dataset_and_tokenizer(self, sequence_length: int) -> tuple[datasets.DatasetDict, character_tokenizer]:
        if not os.path.exists('data/shakespeare_char'):
            os.makedirs('data/shakespeare_char')

        dataset, vocab = None, None

        text = requests.get('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt').text

        vocab = sorted(set(text))

        tokenizer = character_tokenizer(vocab)

        ids = tokenizer.encode(text)

        train_to_val_ratio = 0.9
        train_len = int(len(ids) * train_to_val_ratio)


        dataset = datasets.DatasetDict({
            'train': datasets.Dataset.from_dict({
                'ids': ids[:train_len]
            }),
            'validation': datasets.Dataset.from_dict({
                'ids': ids[train_len:]
            })
        })

        # batch the dataset
        for split in dataset.keys():
            dataset[split] = dataset[split].batch(sequence_length + 1, drop_last_batch = True)

        return dataset.with_format(type = 'torch', columns = ['ids']), tokenizer

    def cache_dataset(self, dataset: datasets.DatasetDict, path: str):
        dataset.save_to_disk(path)

    def save_tokenizer(self, tokenizer: character_tokenizer, path: str):
        # dump the vocab
        with open(path, 'wb') as f:
            for token in tokenizer.vocab:
                f.write(token.encode('utf-8') + b'\n')

    def load_tokenizer(self, path: str) -> character_tokenizer:
        with open(path, 'rb') as f:
            vocab = [line.strip().decode('utf-8') for line in f]

        return character_tokenizer(vocab)

    def encode_text(self, tokenizer: character_tokenizer, text: str) -> list[int]:
        return tokenizer.encode(text)

    def decode_ids(self, tokenizer: character_tokenizer, ids: list[int]) -> str:
        return tokenizer.decode(ids)

    def get_vocab_size(self, tokenizer: character_tokenizer) -> int:
        return tokenizer.get_vocab_size()
