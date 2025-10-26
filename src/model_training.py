import os
import evaluate
import torch # Import torch for device handling
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType # Import PEFT components
from data_preparation import load_and_tokenize_data, MODEL_CHECKPOINT # Import config

# --- Configuration ---
OUTPUT_DIR = "models/fine_tuned_t5_lora" # Changed output directory to distinguish
BATCH_SIZE = 4
LEARNING_RATE = 5e-4
NUM_EPOCHS = 1
SAVE_STRATEGY = "epoch"
LOGGING_STEPS = 50

# LoRA Configuration
LORA_R = 8              # LoRA attention dimension
LORA_ALPHA = 16         # Alpha parameter for LoRA scaling
LORA_DROPOUT = 0.1      # Dropout probability for LoRA layers
LORA_BIAS = "none"      # Can be "none", "all", "lora_only"

# --- Metrics ---
metric = evaluate.load("rouge")

def compute_metrics(eval_pred):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT) # Using base model's tokenizer for consistency
    predictions, labels = eval_pred

    # Ensure labels are a NumPy array if they aren't already, and clone to avoid modifying original
    # This also helps ensure we are working with a mutable array.
    labels = labels.copy() if isinstance(labels, torch.Tensor) else labels

    # Replace -100 in the labels with the tokenizer's pad_token_id
    # This is crucial for successful decoding.
    labels[labels == -100] = tokenizer.pad_token_id

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a list of str for predictions and references
    decoded_preds = ["\n".join(pred.strip().split('\n')) for pred in decoded_preds]
    decoded_labels = ["\n".join(label.strip().split('\n')) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return result

def train_model_with_lora():
    # Ensure dummy data exists for a quick run
    #Ensure the tokenizer and model are configured before attempting to load dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    # Use float32 for CPU explicitly, as bfloat16 is usually GPU-specific.
    # We load the base model here.
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT, torch_dtype=torch.float32)

    # LoRA Configuration
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias=LORA_BIAS,
        target_modules=["q", "v"], # Common target modules for T5
    )

    # Apply LoRA to the model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters() # This will show how few parameters are being trained!

    # Now load and tokenize the actual dataset using the configured tokenizer
    tokenized_datasets = load_and_tokenize_data(tokenizer)
    
    # Use smaller datasets for faster training
    tokenized_datasets["train"] = tokenized_datasets["train"].select(range(100))  # Use only 100 examples
    tokenized_datasets["validation"] = tokenized_datasets["validation"].select(range(20))  # Use only 20 examples

    # Data Collator for Seq2Seq: Dynamically pads the input and target sequences
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        save_strategy=SAVE_STRATEGY,
        save_total_limit=1,
        logging_steps=LOGGING_STEPS,
        eval_strategy="epoch",
        predict_with_generate=True,
        fp16=True, # Enable mixed precision training (if GPU supports it)
        # bf16=True, # If your GPU supports bfloat16, you can use this instead of fp16
        load_best_model_at_end=True,
        metric_for_best_model="rouge1",
        report_to="none",
        # Added for potential QLoRA with 4-bit quantization later (though model is not loaded quantized here)
        gradient_accumulation_steps=4 if BATCH_SIZE == 8 else 1,
        max_grad_norm=1.0, # Example: Increase effective batch size
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting LoRA training...")
    trainer.train()
    print("LoRA training complete!")

    # Save the PEFT model only (the LoRA adapters)
    # The base model is not modified, only the adapters are saved.
    model.save_pretrained(OUTPUT_DIR)
    # The tokenizer also needs to be saved.
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"LoRA adapters and tokenizer saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train_model_with_lora()