import os
import json
from datasets import DatasetDict, load_dataset # Import load_dataset
from transformers import AutoTokenizer

# --- Configuration ---
MODEL_CHECKPOINT = "t5-small"
MAX_INPUT_LENGTH = 512       # Max tokens for input document
MAX_TARGET_LENGTH = 128      # Max tokens for target summary

# --- New Dataset Configuration ---
DATASET_NAME = "abisee/cnn_dailymail"
DATASET_VERSION = "2.0.0" # Specify the version. "3.0.0" is another common one.
                         # Version "2.0.0" is pre-tokenized and often faster for T5-like models.
                         # Version "3.0.0" is the full text. Let's use 2.0.0 for simplicity here.
                         # If you want to use 3.0.0, you might need to adjust column names.

def load_and_tokenize_data(tokenizer):
    """Loads, processes, and tokenizes the cnn_dailymail dataset."""

    print(f"Loading dataset: {DATASET_NAME}, version: {DATASET_VERSION}")
    # Load the dataset from Hugging Face Hub
    # Choose specific splits for efficiency. 'train', 'validation', 'test' are standard.
    raw_datasets = load_dataset(DATASET_NAME, DATASET_VERSION, split=['train', 'validation', 'test'])

    # The split returns a list of datasets, convert it to a DatasetDict
    raw_datasets = DatasetDict({
        "train": raw_datasets[0],
        "validation": raw_datasets[1],
        "test": raw_datasets[2]
    })

    print("Raw datasets loaded:")
    print(raw_datasets)
    print(f"Example features: {raw_datasets['train'].column_names}")

    # Identify the correct columns for text and summary
    # For cnn_dailymail v2.0.0, it's typically 'article' and 'highlights'
    # For v3.0.0, it might also be 'article' and 'highlights' or similar.
    # Always print `raw_datasets['train'].column_names` to verify!
    text_column = "article"
    summary_column = "highlights"


    def preprocess_function(examples):
        inputs = [doc for doc in examples[text_column]]
        model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples[summary_column], max_length=MAX_TARGET_LENGTH, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        # Remove original text and summary columns, keep only tokenized inputs and labels
        remove_columns=raw_datasets["train"].column_names,
        load_from_cache_file=True, # Use cache for faster re-runs
        desc="Running tokenizer on dataset"
    )

    print("\nTokenized datasets:")
    print(tokenized_datasets)
    return tokenized_datasets

if __name__ == "__main__":
    # Removed create_dummy_data() call
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    tokenized_datasets = load_and_tokenize_data(tokenizer)

    # Note: For very large datasets, you might want to slice them for faster testing
    # For example, to use only 1000 examples from train and 100 from validation/test:
    # tokenized_datasets["train"] = tokenized_datasets["train"].select(range(1000))
    # tokenized_datasets["validation"] = tokenized_datasets["validation"].select(range(100))
    # tokenized_datasets["test"] = tokenized_datasets["test"].select(range(100))
    # print("Sliced tokenized datasets for quick testing:")
    # print(tokenized_datasets)

    print("Data preparation complete!")