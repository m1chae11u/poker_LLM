import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import json
from typing import Any, Dict, List

class CustomSFTDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        
        instruction = example["instruction"]
        output_text = example["output"]

        # Combine instruction and output
        full_text = instruction + "\n" + output_text

        # Tokenize without converting to tensors
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        # Create labels
        labels = input_ids.copy()

        # Mask instruction tokens
        instruction_ids = self.tokenizer(
            instruction,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]
        instruction_length = len(instruction_ids)
        labels[:instruction_length] = [-100] * instruction_length

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

def load_datasets(data_path_list):
    data = []
    for path in data_path_list:
        with open(path, 'r') as f:
            data.extend(json.load(f))
    return data

def custom_data_collator(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    batch = {}

    max_length = max(len(feature["input_ids"]) for feature in features)

    input_ids_batch = []
    attention_mask_batch = []
    labels_batch = []

    for feature in features:
        input_ids = feature["input_ids"]
        attention_mask = feature["attention_mask"]
        labels = feature["labels"]

        # Pad input_ids and attention_mask
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
        attention_mask = attention_mask + [0] * padding_length

        # Pad labels with -100
        labels = labels + [-100] * padding_length

        input_ids_batch.append(input_ids)
        attention_mask_batch.append(attention_mask)
        labels_batch.append(labels)

    batch["input_ids"] = torch.tensor(input_ids_batch, dtype=torch.long)
    batch["attention_mask"] = torch.tensor(attention_mask_batch, dtype=torch.long)
    batch["labels"] = torch.tensor(labels_batch, dtype=torch.long)

    return batch

if __name__ == "__main__":
    train_data_paths = [
        "/home/michael_lu/poker_LLM/data/postflop_500k_train_set_25252525.json",
        "/home/michael_lu/poker_LLM/data/preflop_60k_train_set.json"
    ]
    
    test_data_paths = [
        "/home/michael_lu/poker_LLM/data/postflop_10k_test_set.json",
        "/home/michael_lu/poker_LLM/data/preflop_1k_test_set.json"
    ]

    # Load datasets separately
    train_data = load_datasets(train_data_paths)
    test_data = load_datasets(test_data_paths)

    model_name = "/data/akshat/models/Llama-2-7b-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )

    train_dataset = CustomSFTDataset(data=train_data, tokenizer=tokenizer, max_length=512)
    eval_dataset = CustomSFTDataset(data=test_data, tokenizer=tokenizer, max_length=512)

    training_args = TrainingArguments(
        output_dir="./llama2_poker_sft",
        overwrite_output_dir=True,
        per_device_train_batch_size=2,
        num_train_epochs=3,
        save_steps=500,
        logging_steps=100,
        save_total_limit=2,
        evaluation_strategy="steps",
        eval_steps=500,
        report_to="none",
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=custom_data_collator,  # Use the custom data collator
    )

    trainer.train()
    trainer.save_model()