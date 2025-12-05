import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import LoraConfig, get_peft_model, TaskType
import os

# --- Configuration ---
MODEL_NAME = "google/flan-t5-large"
DATA_PATH = "Blooms-Followup-Questions/BloomsFQ.csv"
OUTPUT_DIR = "bfqg_finetuned"
MAX_SOURCE_LENGTH = 256
MAX_TARGET_LENGTH = 128
BATCH_SIZE = 4
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3

# Bloom's Taxonomy Levels Mapping (matching bfqg_generator.py)
BLOOMS_LEVELS = {
    "FQ1": "Understand (Explain concepts)",
    "FQ2": "Apply (Use in new situation)",
    "FQ3": "Analyze (Break down/compare)",
    "FQ4": "Evaluate (Judge/critique)",
    "FQ5": "Create (Design/propose new)",
}


def prepare_data():
    """Load and preprocess the CSV data into prompt-completion pairs."""
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    data_entries = []

    for _, row in df.iterrows():
        seed_question = row["Seed Question"]

        for col, level_desc in BLOOMS_LEVELS.items():
            if pd.notna(row[col]):
                target_question = row[col]

                # Construct prompt matching the generator's format (without few-shot examples)
                prompt = f"""Task: Generate a single follow-up question for an in-car assistant based on the seed question.

Now, generate a question for the following new task:
Seed: "{seed_question}"
Target Level: {level_desc}
Constraint: The question must strictly adhere to the target level and the specific seed topic. Output ONLY the question text.
Question:"""

                data_entries.append({"input_text": prompt, "target_text": target_question})

    print(f"Created {len(data_entries)} training examples.")
    return Dataset.from_list(data_entries)


def preprocess_function(examples, tokenizer):
    """Tokenize inputs and targets."""
    model_inputs = tokenizer(examples["input_text"], max_length=MAX_SOURCE_LENGTH, truncation=True, padding="max_length")

    labels = tokenizer(examples["target_text"], max_length=MAX_TARGET_LENGTH, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    # 1. Prepare Data
    dataset = prepare_data()

    # 2. Split Data
    dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # 3. Load Model and Tokenizer
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # Preprocess datasets
    tokenized_train = train_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True, remove_columns=train_dataset.column_names)
    tokenized_eval = eval_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True, remove_columns=eval_dataset.column_names)

    # 4. Setup PEFT (LoRA)
    print("Setting up LoRA...")
    peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 5. Training Arguments
    # Detect device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Training on device: {device}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=8,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=10,
        use_mps_device=(device == "mps"),
        predict_with_generate=True,
        save_total_limit=2,
        load_best_model_at_end=True,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,  # Ignore padding in loss calculation
    )

    # 6. Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 7. Train
    print("Starting training...")
    trainer.train()

    # 8. Save Model
    print(f"Saving model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done!")


if __name__ == "__main__":
    main()
