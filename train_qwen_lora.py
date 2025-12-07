import os
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training

# ------------------------
# CONFIG
# ------------------------
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DATA_FOLDER = "data"
OUTPUT_DIR = "adapter_qwen"
MAX_LENGTH = 1024
BATCH_SIZE = 1
EPOCHS = 2

# ------------------------
# TOKENIZER
# ------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ------------------------
# MODEL
# ------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype="auto",
)

# Przy małym VRAM odblokowujemy LoRA + zamrażamy resztę
model = prepare_model_for_int8_training(model)

# ------------------------
# LoRA
# ------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# ------------------------
# DATASET
# ------------------------
# łączymy wszystkie pliki .txt w folderze data
data_files = [os.path.join(DATA_FOLDER, f) for f in os.listdir(DATA_FOLDER) if f.endswith(".txt")]
dataset = load_dataset("text", data_files={"train": data_files})

# tokenizacja + labels
def tokenize_function(examples):
    tokens = tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)
    tokens["labels"] = tokens["input_ids"].copy()  # konieczne dla obliczenia loss
    return tokens

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# ------------------------
# TRAINING ARGUMENTS
# ------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=1,
    warmup_steps=10,
    num_train_epochs=EPOCHS,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    fp16=True,
    gradient_checkpointing=True,
)

# ------------------------
# TRAINER
# ------------------------
trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset["train"],
    args=training_args,
)

# ------------------------
# START TRAINING
# ------------------------
trainer.train()
