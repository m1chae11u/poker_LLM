import json
from pathlib import Path

def remove_binary_indicator_from_directory(directory_path):
    # Get all JSON files in the specified directory
    json_files = Path(directory_path).glob("*.json")
    
    for json_file in json_files:
        # Load the JSON data
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Remove "binary_indicator" field from each entry if it exists
        for entry in data:
            if "binary_indicator" in entry:
                del entry["binary_indicator"]
        
        # Save the modified data back to the original file
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"Processed and saved: {json_file}")

if __name__ == "__main__":
    directory_path = "/home/michael_lu/poker_LLM/data/formatted_for_lit_gpt" 
    remove_binary_indicator_from_directory(directory_path)