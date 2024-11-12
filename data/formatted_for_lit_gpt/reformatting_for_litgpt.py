import json
import os

def combine_datasets(train_files, val_files, output_dir=".", output_train="train.json", output_val="val.json"):
    """
    Combine multiple JSON files into a single training and validation file.
    
    Parameters:
    - train_files: List of file paths for training data
    - val_files: List of file paths for validation data
    - output_dir: Directory where the combined JSON output files will be saved
    - output_train: Name of the combined training JSON output file
    - output_val: Name of the combined validation JSON output file
    """
    combined_train_data = []
    combined_val_data = []
    
    # Load and combine all training data files
    for file_path in train_files:
        with open(file_path, 'r') as file:
            data = json.load(file)
            combined_train_data.extend(data)
    
    # Load and combine all validation data files
    for file_path in val_files:
        with open(file_path, 'r') as file:
            data = json.load(file)
            combined_val_data.extend(data)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output file paths
    output_train_path = os.path.join(output_dir, output_train)
    output_val_path = os.path.join(output_dir, output_val)
    
    # Write combined data to output files
    with open(output_train_path, 'w') as file:
        json.dump(combined_train_data, file, indent=4)
    
    with open(output_val_path, 'w') as file:
        json.dump(combined_val_data, file, indent=4)
    
    print(f"Combined training data saved to {output_train_path}")
    print(f"Combined validation data saved to {output_val_path}")

if __name__ == "__main__":
    # Define your file paths
    train_files = [
        "postflop_500k_train_set_25252525_with_input.json",
        "preflop_60k_train_set_with_input.json"
    ]

    val_files = [
        "postflop_10k_test_set_with_input.json",
        "preflop_1k_test_set_with_input.json"
    ]

    # Combine datasets and specify an output directory
    combine_datasets(train_files, val_files, output_dir="/home/michael_lu/poker_LLM/data/formatted_for_lit_gpt/lit_gpt_custom_setup")