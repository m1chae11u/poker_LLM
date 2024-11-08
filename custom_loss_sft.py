# custom_loss_sft.py

import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
from torch.nn import CrossEntropyLoss

class CustomSFTDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length=512, mask_function=None):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_function = mask_function if mask_function else self.default_mask_function

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        
        instruction = example["instruction"]
        output_text = example["output"]
        
        # Encode the full text
        encoded = self.tokenizer.encode_plus(
            instruction + output_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].squeeze()
        attention_mask = encoded["attention_mask"].squeeze()
        
        labels = input_ids.clone()
        
        # Mask the instruction part
        instruction_length = len(self.tokenizer.encode(instruction, add_special_tokens=False))
        labels[:instruction_length] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def default_mask_function(self, input_ids, output_ids, labels):
        prompt_length = len(input_ids)
        labels[:prompt_length] = -100
        return labels

class CustomSFTTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        attention_mask = inputs.get("attention_mask")
        outputs = model(input_ids=inputs.get("input_ids"), attention_mask=attention_mask)
        logits = outputs.get("logits")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return (loss, outputs) if return_outputs else loss

def load_data(file_paths):
    data = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            data.extend(json.load(f))
    return data

if __name__ == "__main__":
    train_data_paths = [
        "/home/michael_lu/poker_LLM/data/postflop_500k_train_set_25252525.json",
        "/home/michael_lu/poker_LLM/data/preflop_60k_train_set.json"
    ]
    
    test_data_paths = [
        "/home/michael_lu/poker_LLM/data/postflop_10k_test_set.json",
        "/home/michael_lu/poker_LLM/data/preflop_1k_test_set.json"
    ]
    
    # Load the data
    train_data = load_data(train_data_paths)
    test_data = load_data(test_data_paths)

    # Initialize model and tokenizer
    # model_name = "google/gemma-2-2b"
    model_name = "/data/akshat/models/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    # Create datasets
    train_dataset = CustomSFTDataset(data=train_data, tokenizer=tokenizer, max_length=512)
    test_dataset = CustomSFTDataset(data=test_data, tokenizer=tokenizer, max_length=512)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./custom_loss_sft_results_test",
        overwrite_output_dir=True,
        per_device_train_batch_size=2,
        num_train_epochs=3,
        save_steps=10,
        logging_steps=5,
        report_to="none",
    )

    # Initialize trainer
    trainer = CustomSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )

    # Start training
    trainer.train()
    trainer.save_model()
