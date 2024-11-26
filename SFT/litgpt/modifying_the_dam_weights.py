import json

# Path to the input JSON file
input_path = "/data/michael_lu/poker_LLM/litgpt/checkpoints/google/gemma-2b/model.safetensors.index.json"

# Path to save the updated JSON file
output_path = "/data/michael_lu/poker_LLM/litgpt/checkpoints/google/gemma-2b/updated_model.safetensors.index.json"

# Load the current weight map
with open(input_path, "r") as f:
    data = json.load(f)

# Access the weight_map dictionary
weight_map = data["weight_map"]

# Create a new weight map with updated keys
updated_weight_map = {}

for key, value in weight_map.items():
    updated_key = key

    # Replace 'post_attention_layernorm' with 'norm_2'
    if "post_attention_layernorm" in key:
        updated_key = updated_key.replace("post_attention_layernorm", "norm_2")

    # Replace 'model.layers' with 'transformer.h'
    if "model.layers" in key:
        updated_key = updated_key.replace("model.layers", "transformer.h")

    # Add the updated key-value pair to the new map
    updated_weight_map[updated_key] = value

# Update the data with the new weight map
data["weight_map"] = updated_weight_map

# Save the updated JSON back to the output file
with open(output_path, "w") as f:
    json.dump(data, f, indent=4)

print(f"Updated weight map saved to {output_path}")