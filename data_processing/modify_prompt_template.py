# modify_prompt_template.py

import os
import json
import tempfile
import shutil

def modify_prompt_template(dir_path, section_to_modify, new_section_replacement):
    """
    Scans the directory for JSON files, modifies a specific part of the 'instruction' field,
    and writes the updated JSON back to the file using a safe save method.

    Args:
        dir_path (str): The directory path containing the JSON files.
        section_to_modify (str): The exact text to locate and modify within the 'instruction'.
        new_section_replacement (str): The replacement text for the specified section.

    Returns:
        List[dict]: A list of modified JSON objects with their file locations.
    """
    modified_files = []

    # List of specific files to modify
    files_to_modify = {
        "preflop_60k_train_set.json",
        "postflop_500k_train_set_25252525.json",
        "postflop_10k_test_set.json",
        "preflop_1k_test_set.json"
    }

    # Scan the directory
    for file_name in os.listdir(dir_path):
        if file_name in files_to_modify:  # Only process specified files
            file_path = os.path.join(dir_path, file_name)  # Full path to the file
            try:
                # Load the JSON content
                with open(file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)  # Load JSON as Python objects
                
                modified = False

                # Check and modify each object in the JSON array
                for obj in data:
                    if "instruction" in obj and section_to_modify in obj["instruction"]:
                        # Replace the target section in 'instruction'
                        obj["instruction"] = obj["instruction"].replace(section_to_modify, new_section_replacement)
                        modified = True

                # If modification occurred, write back using a safe saving method
                if modified:
                    # Create a temporary file for safe saving
                    temp_file = tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8")
                    try:
                        # Save the modified data to the temporary file
                        json.dump(data, temp_file, indent=4, ensure_ascii=False)
                        temp_file.close()
                        # Replace the original file with the temporary file
                        shutil.move(temp_file.name, file_path)
                        modified_files.append({"file_path": file_path, "modified_content": data})
                    except Exception as e:
                        print(f"Failed to save changes to {file_path}: {e}")
                        # Ensure temp file is removed if there's an error
                        os.unlink(temp_file.name)

            except (json.JSONDecodeError, IOError) as e:
                print(f"Error processing file {file_name}: {e}")

    return modified_files


if __name__ == "__main__":
    dir_path = "/data/sergio_peterson/data"
    section_to_modify = "Do not explain your answer.\nYour optimal action is:"
    new_section_replacement = "Explain your answer and output your optimal action."

    modified_files = modify_prompt_template(dir_path, section_to_modify, new_section_replacement)
    if modified_files:
        for file_info in modified_files:
            print(f"Modified file: {file_info['file_path']}")
    else:
        print("No modifications were made.")