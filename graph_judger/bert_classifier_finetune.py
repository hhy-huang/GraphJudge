import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer
)

# Training hyperparameters
MICRO_BATCH_SIZE = 16
BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 4
LEARNING_RATE = 2e-5
CUTOFF_LEN = 512
VAL_SET_SIZE = 2000

# Paths
DATA_PATH = "data/rebel_sub_4omini_context/train_instructions_context_llama2_7b.json"
OUTPUT_DIR = "models/bert-base-classifier-rebel-sub"
base_model_path = "google-bert/bert-base-uncased"

# DDP setup
device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained(base_model_path)
model = BertForSequenceClassification.from_pretrained(
    base_model_path,
    num_labels=2,  # Binary classification: correct or incorrect
    problem_type="single_label_classification"
)

def generate_and_tokenize_prompt(data_point):
    # Convert inputs to strings
    instruction = str(data_point["instruction"])
    input_text = str(data_point["input"]) if data_point["input"] else ""
    
    # Combine instruction and input
    if input_text:
        text = f"Instruction: {instruction} Input: {input_text}"
    else:
        text = f"Instruction: {instruction}"
    
    # Convert output to binary label (assuming 'output' contains 'true'/'false' or similar)
    # Adjust this part according to your actual output format
    label = 1 if str(data_point["output"]).lower() in ['true', 'correct', 'yes'] else 0
    
    # Tokenize
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding="max_length",
        return_tensors=None,
    )
    
    # Add labels to the encoded dict
    encoded["labels"] = label
    return encoded

if __name__ == "__main__":
    # Load dataset
    data = load_dataset("json", data_files=DATA_PATH)
    
    # Data preprocessing
    if VAL_SET_SIZE > 0:
        train_val = data["train"].train_test_split(
            test_size=VAL_SET_SIZE, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"]
            .shuffle()
            .map(generate_and_tokenize_prompt, remove_columns=data["train"].column_names)
        )
        val_data = (
            train_val["test"]
            .shuffle()
            .map(generate_and_tokenize_prompt, remove_columns=data["train"].column_names)
        )
    else:
        train_data = (
            data["train"]
            .shuffle()
            .map(generate_and_tokenize_prompt, remove_columns=data["train"].column_names)
        )
        val_data = None

    # Initialize trainer
    training_args = TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=5,
        evaluation_strategy="steps" if VAL_SET_SIZE > 0 else "no",
        save_strategy="steps",
        eval_steps=20 if VAL_SET_SIZE > 0 else None,
        save_steps=20,
        output_dir=OUTPUT_DIR,
        save_total_limit=3,
        load_best_model_at_end=True if VAL_SET_SIZE > 0 else False,
        ddp_find_unused_parameters=False if ddp else None,
        report_to="wandb",
        logging_strategy="steps",
        logging_first_step=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
    )

    # Train the model
    trainer.train()

    # Save model and tokenizer
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("\nTraining completed! Model saved to:", OUTPUT_DIR) 