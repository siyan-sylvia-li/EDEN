import os
import torch
from datasets import load_dataset
import datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

from utils import find_all_linear_names, print_trainable_parameters
import pandas as pd

# Used for multi-gpu
local_rank = -1
per_device_train_batch_size = 4
per_device_eval_batch_size = 4
gradient_accumulation_steps = 1
learning_rate = 2e-4
max_grad_norm = 0.3
weight_decay = 0.001
lora_alpha = 16
lora_dropout = 0.1
lora_r = 64
max_seq_length = None

# The model that you want to train from the Hugging Face hub
model_name = "meta-llama/Llama-2-7b-chat-hf"#"guardrail/llama-2-7b-guanaco-instruct-sharded"

# Fine-tuned model name
new_model = "llama-2-7b-grammar"

# Activate 4-bit precision base model loading
use_4bit = True

# Activate nested quantization for 4-bit base models
use_nested_quant = False

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Number of training epochs
num_train_epochs = 3

# Enable fp16 training, (bf16 to True with an A100)
fp16 = False

# Enable bf16 training
bf16 = False

# Use packing dataset creating
packing = False

# Enable gradient checkpointing
gradient_checkpointing = True

# Optimizer to use, original is paged_adamw_32bit
optim = "paged_adamw_32bit"

# Learning rate schedule (constant a bit better than cosine, and has advantage for analysis)
lr_scheduler_type = "cosine"

# Number of optimizer update steps, 10K original, 20 for demo purposes
max_steps = -1

# Fraction of steps to do a warmup for
warmup_ratio = 0.03

# Group sequences into batches with same length (saves memory and speeds up training considerably)
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 200

# Log every X updates steps
logging_steps = 50

# The output directory where the model predictions and checkpoints will be written
output_dir = "./results"

# Load the entire model on the GPU 0
device_map = {"": 0}

# Visualize training
report_to = "tensorboard"

# Tensorboard logs
tb_log_dir = "./results/logs"


def load_model(model_name):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                                      quantization_config=bnb_config)
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(base_model)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    # Change the LORA hyperparameters accordingly to fit your use case
    peft_config = LoraConfig(
        r=128,
        lora_alpha=16,
        target_modules=find_all_linear_names(base_model),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    base_model = get_peft_model(base_model, peft_config)
    print_trainable_parameters(base_model)

    return base_model, tokenizer, peft_config


# template dataset to add prompt to each sample
def template_dataset(sample):
    instruction = f"<s>[INST] {sample['prompt']}"
    response = f" [/INST] {sample['response']}"
    # join all the parts together
    prompt = "".join([i for i in [instruction, response] if i is not None])
    sample["text"] = f"{prompt}{tokenizer.eos_token}"
    return sample


if __name__ == "__main__":
    model, tokenizer, peft_config = load_model(model_name)
    # apply prompt template per sample
    DATA_PATH = "data/"
    training_data = pd.read_csv(os.path.join(DATA_PATH, "train_llama.csv"))
    eval_data = pd.read_csv(os.path.join(DATA_PATH, "val_llama.csv"))
    dataset_train = datasets.Dataset.from_pandas(training_data)
    dataset_val = datasets.Dataset.from_pandas(eval_data)

    dataset_train = dataset_train.map(template_dataset, remove_columns=list(dataset_train.features))
    dataset_val = dataset_val.map(template_dataset, remove_columns=list(dataset_val.features))

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to="tensorboard",
        num_train_epochs=num_train_epochs,
        evaluation_strategy="steps",
        save_total_limit=3,
        save_strategy="steps",
        load_best_model_at_end=True
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
    )

    # trainer.train(resume_from_checkpoint=True)
    trainer.train()
    trainer.save_model(output_dir)

    output_dir = os.path.join(output_dir, "final_checkpoint_llama")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

