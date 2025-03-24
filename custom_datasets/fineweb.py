import transformers
import datasets


def get_dataset_and_tokenizer(sequence_length: int):
    # load fineweb 10B dataset
    dataset = datasets.load_dataset("HuggingFaceFW/fineweb-edu", name = "sample-10BT", split = "train", streaming = True)

    # take only 2B tokens of the dataset
    dataset = dataset.shard(num_shards=5, index=0)

    tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding = "max_length",
            truncation = True,
            padding_side = "right",
            max_length = sequence_length + 1,
            return_tensors = "pt"
        )
    

    dataset = dataset.map(
        tokenize_function,
        remove_columns = ["text"]
    )

    # Create train/validation split for streaming
    train_dataset = dataset.filter(lambda x, i: i % 100 != 0, with_indices=True)
    val_dataset = dataset.filter(lambda x, i: i % 100 == 0, with_indices=True)

    # No need to set format as it's already handled in the tokenize_function
    return {"train": train_dataset, "validation": val_dataset}, tokenizer


    