import json
from pathlib import Path

def insert_empty_input(filepath):
    # Load the JSON data
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Insert an empty "input" field after "instruction" and reorder fields
    for entry in data:
        if "instruction" in entry and "output" in entry and "binary_indicator" in entry:
            entry["input"] = ""  # Temporarily add input
            # Reorder fields to the specified order
            reordered_entry = {
                "instruction": entry["instruction"],
                "input": entry["input"],
                "output": entry["output"],
                "binary_indicator": entry["binary_indicator"]
            }
            # Update the entry with reordered fields
            entry.clear()
            entry.update(reordered_entry)
    
    # Save the modified data to a new file
    new_filepath = Path(filepath).with_name(f"{Path(filepath).stem}_with_input.json")
    with open(new_filepath, 'w') as f:
        json.dump(data, f, indent=4)
    
    return new_filepath

if __name__ == "__main__":
    dataset_files = [
        "/home/michael_lu/poker_LLM/data/postflop_10k_test_set.json",
        "/home/michael_lu/poker_LLM/data/postflop_500k_train_set_25252525.json",
        "/home/michael_lu/poker_LLM/data/preflop_1k_test_set.json",
        "/home/michael_lu/poker_LLM/data/preflop_60k_train_set.json"
    ]
    
    # Process each file to insert empty "input" fields and save as new files
    for file in dataset_files:
        new_filepath = insert_empty_input(file)
        print(f"Processed and saved: {new_filepath}")