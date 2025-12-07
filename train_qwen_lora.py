from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import torch


model_name = "Qwen/Qwen3-4B-Instruct-2507"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)


lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)


dataset = load_dataset("text", data_files={"train": "data/*.txt"})

def tokenize_function(examples):
    tokens = tokenizer(examples["text"], truncation=True, max_length=1024)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize_function, batched=True)


training_args = TrainingArguments(
    output_dir="adapter_qwen",
    per_device_train_batch_size=1,
    num_train_epochs=2,
    logging_steps=50,
    save_steps=500,
    fp16=True,
    gradient_checkpointing=True,
    save_total_limit=2
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"]
)


trainer.train()


model.save_pretrained("adapter_qwen")
tokenizer.save_pretrained("adapter_qwen")
