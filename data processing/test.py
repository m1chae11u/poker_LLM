# test.py

import pandas as pd
import os
import json
from transformers import AutoTokenizer

dir_of_this_script = os.path.dirname(os.path.abspath(__file__))
path_to_config = os.path.join(dir_of_this_script, 'configs', 'config.json')
with open(path_to_config, 'r') as config_file:
    config_data = json.load(config_file)
os.environ['HF_TOKEN'] = config_data["HF_TOKEN"]

# Import your updated functions (assuming they are in a file named 'poker_data_processor.py')
# If the functions are in the same file, you can skip the import statement
# from poker_data_processor import poker_csv_to_json, initialize_tokenizer

# For this test, we'll include the necessary parts of the code directly

# Initialize the tokenizer
def initialize_tokenizer(model_name=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def test_tokenization():
    # Sample data
    sample_data = {
        'hero_pos': ['CO'],
        'hero_holding': ['9d9c'],
        'prev_line': ['UTG/2.0/CO/call/BTN/call/BB/14.0/UTG/fold/CO/call/BTN/allin/BB/fold'],
        'pot_size': [130.5],
        'available_moves': ['["fold", "call", "raise 18.5bb"]'],
        'correct_decision': ['fold'],
        'hand_strength': ['medium'],
        'evaluation_at': ['Flop'],
        'preflop_action': ['UTG/2.0/CO/call'],
        'postflop_action': [''],
        'board_flop': ['2hTsAc'],
        'board_turn': [''],
        'board_river': [''],
        'holding': ['9d9c'],
        'hero_position': ['CO']
    }

    # Create DataFrame
    preflop_df = pd.DataFrame(sample_data)
    postflop_df = pd.DataFrame(sample_data)

    # Replace the placeholder function in the main code
    global sergio_custom_function

    # Define the sergio_custom_function placeholder
    def sergio_custom_function():
        return 'one pair (a pair of nines)'

    # Include the construct_prompt function with necessary modifications
    def construct_prompt_preflop(row: pd.Series):
        # Parse dynamic parts
        preflop_action_summary = row['prev_line']
        hero_position = row['hero_pos']
        hero_holding = row['hero_holding']
        current_pot_size = row['pot_size']
        best_current_hand = sergio_custom_function()

        # Build the prompt using segments with dynamic indicators
        segments = []

        # Static segments
        segments.append(("You are a specialist in playing 6-handed No Limit Texas Holdem. The following will be a game scenario and you need to make the optimal decision.\n\nHere is a game summary:\n\n", 0))
        segments.append(("The small blind is 0.5 chips and the big blind is 1 chips. Everyone started with 100 chips.\nThe player positions involved in this game are UTG, HJ, CO, BTN, SB, BB.\n", 0))
        segments.append(("In this hand, your position is ", 0))

        # Dynamic segment: hero_position
        segments.append((hero_position, 1))
        segments.append((", and your holding is ", 0))

        # Dynamic segment: hero_holding
        segments.append((hero_holding, 1))
        segments.append((".\n", 0))

        # Dynamic segment: best_current_hand, hand_strength
        segments.append(("You currently have ", 0))
        segments.append((best_current_hand, 1))
        segments.append((".\n", 0))

        segments.append(("Before the flop, ", 0))

        # Dynamic segment: preflop_action_summary
        segments.append((preflop_action_summary, 1))
        segments.append((". Assume that all other players that is not mentioned folded.\n\nNow it is your turn to make a move.\nTo remind you, the current pot size is ", 0))

        # Dynamic segment: current_pot_size
        segments.append((str(current_pot_size), 1))
        segments.append((" chips, and your holding is ", 0))

        # Dynamic segment: hero_holding
        segments.append((hero_holding, 1))
        segments.append((".\n\nDecide on an action based on the strength of your hand on this board, your position, and actions before you. Do not explain your answer.\nYour optimal action is:", 0))

        # Build the full prompt and per-character dynamic indicators
        prompt_text = ''
        dynamic_char_indicators = []
        for segment in segments:
            text, is_dynamic = segment
            prompt_text += text
            dynamic_char_indicators.extend([is_dynamic]*len(text))

        # Tokenize the prompt
        model_name = "meta-llama/meta-llama-3.1-70b-instruct"
        tokenizer = initialize_tokenizer(model_name=model_name)
        encoding = tokenizer(prompt_text, return_offsets_mapping=True, add_special_tokens=False)
        tokens = encoding.tokens()
        offsets = encoding.offset_mapping

        # Build the binary_indicator
        binary_indicator = []
        for (start, end) in offsets:
            # Check the dynamic indicators for this token's span
            token_dynamic_chars = dynamic_char_indicators[start:end]
            if any(token_dynamic_chars):
                binary_indicator.append(1)
            else:
                binary_indicator.append(0)

        # Process the correct decision
        correct_decision = row['correct_decision'].lower()

        # Return the prompt, correct decision, and binary_indicator
        return prompt_text, correct_decision, binary_indicator, tokens

    # Run the test for preflop
    print("Testing Preflop Tokenization and Binary Indicators:")
    for index, row in preflop_df.iterrows():
        prompt_text, correct_decision, binary_indicator, tokens = construct_prompt_preflop(row)
        print("\nPrompt Text:\n", prompt_text)
        print("\nTokens and Binary Indicators:")
        for token, indicator in zip(tokens, binary_indicator):
            print(f"Token: {token}, Binary Indicator: {indicator}")
        print("\nCorrect Decision:", correct_decision)

        print(binary_indicator)

    # Similarly, you can define and test the postflop construct_prompt function
    # For brevity, let's proceed to test only the preflop scenario

if __name__ == "__main__":
    test_tokenization()