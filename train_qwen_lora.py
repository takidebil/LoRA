import os
from datasets import load_dataset
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# -----------------------------
# CONFIG
# -----------------------------
model_name = "Qwen/Qwen3-4B-Instruct-2507"
data_path = "data/"
output_dir = "adapter_qwen_lora"

# -----------------------------
# DATASETS LOADING
# -----------------------------
files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".txt")]
dataset = load_dataset("text", data_files={"train": files})
dataset = dataset["train"].train_test_split(test_size=0.1)
dataset = dataset.map(lambda x: {"text": x["text"]})

# -----------------------------
# TOKENIZER
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    tokens = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# -----------------------------
# LOSS FUNCTION
# -----------------------------
loss_fct = nn.CrossEntropyLoss()

# -----------------------------
# MODEL LOADING
# -----------------------------
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_8bit=True,
    torch_dtype=torch.float16
)
model = prepare_model_for_kbit_training(model)

model.gradient_checkpointing_enable()
model.config.use_cache = False

# -----------------------------
# LoRA CONFIG
# -----------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# -----------------------------
# TRAINING ARGUMENTS
# -----------------------------
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    save_steps=500,
    save_total_limit=2,
    logging_steps=50,
    learning_rate=2e-4,
    fp16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    report_to="none"
)

# -----------------------------
# CUSTOM TRAINER
# -----------------------------
class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return (loss, outputs) if return_outputs else loss

trainer = MyTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer
)

# -----------------------------
# TRAIN START
# -----------------------------
trainer.train()
trainer.save_model(output_dir)

print("Adapter zapisany w:", output_dir)
