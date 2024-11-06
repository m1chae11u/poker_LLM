import torch
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
        input_text = example["input"]
        output_text = example["output"]

        instruction_ids = self.tokenizer.encode(instruction, add_special_tokens=False)
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
        output_ids = self.tokenizer.encode(output_text, add_special_tokens=False)
        
        input_ids_full = instruction_ids + input_ids + output_ids
        if len(input_ids_full) > self.max_length:
            input_ids_full = input_ids_full[:self.max_length]

        labels = input_ids_full.copy()

        labels = self.mask_function(instruction_ids, input_ids, output_ids, labels)

        input_ids_full = torch.tensor(input_ids_full, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        
        return {"input_ids": input_ids_full, "labels": labels}

    def default_mask_function(self, instruction_ids, input_ids, output_ids, labels):
        prompt_length = len(instruction_ids) + len(input_ids)
        labels[:prompt_length] = [-100] * prompt_length
        return labels

class CustomSFTTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(input_ids=inputs.get("input_ids"))
        logits = outputs.get("logits")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return (loss, outputs) if return_outputs else loss

# data = [
#     {"instruction": "Describe the scene.", "input": "It's a sunny day in the park.", "output": "The park is filled with people enjoying the sun."},
# ]

model_name = "Qwen/Qwen2-0.5B"
# model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

train_dataset = CustomSFTDataset(data=data, tokenizer=tokenizer, max_length=512)
training_args = TrainingArguments(
    output_dir="./custom_loss_sft_results_test",
    overwrite_output_dir=True,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_steps=10,
    logging_steps=5,
    report_to="none",
)

trainer = CustomSFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model()