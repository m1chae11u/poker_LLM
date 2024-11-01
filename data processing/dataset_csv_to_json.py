import pandas as pd
import ast,json,random
from transformers import AutoTokenizer

def initialize_tokenizer(model_name=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def sergio_custom_function(): # PLACEHOLDER, REPLACE WITH ACTUAL FUNCTION
    current_best_hand = "one pair"
    hand_description = "pair of nines"
    return f" {current_best_hand} (a {hand_description})"

random.seed(42)
def preflop_csv_to_json(preflop_dataset: pd.DataFrame):
    # TODO: No need to parse card now.
    def parse_action(line):
        if line == "":
            return "there has been no action yet"
        parts = line.split('/')
        description = []
        
        # Helper function to get the verbal action
        def get_action(index, bet):
            if 'bb' in bet:
                return f"{parts[index]} raise {bet.split('bb')[0]}"
            elif 'allin' in bet:
                return f"{parts[index]} all in"
            elif 'fold' in bet:
                return f"{parts[index]} fold"
            elif 'call' in bet:
                return f"{parts[index]} call"
            elif 'check' in bet:
                return f"{parts[index]} check"
            elif 'bet' in bet:
                amount = bet.replace('bet', '').strip()
                return f"{parts[index]} bet {amount}"
            else:
                return f"{parts[index]} {bet}"
        
        # Iterate over parts and construct the description list
        i = 0
        while i < len(parts):
            if i+1 < len(parts):
                description.append(get_action(i, parts[i+1]))
            i += 2
        
        # Properly join all elements with commas, and 'and' for the last element
        if len(description) > 1:
            # Join all elements with a comma except the last two
            result = ', '.join(description[:-1])
            # Add 'and' before the last element
            result += f", and {description[-1]}"
        else:
            result = description[0] if description else ''
        
        return result
    # def parse_holding(hand):
    #     suits = ["Spade", 'Heart', "Club", "Diamond"]
    #     rank_dict = {"2": "Two", "3": "Three", "4": "Four", "5":"Five", "6": "Six", "7": "Seven",
    #                 "8": "Eight", "9": "Nine", "T": "Ten", "J":"Jack", "Q": "Queen", "K": "King", "A": "Ace"}
    #     suit_sample = random.sample(suits, 2)
    #     if len(hand) == 2:
    #         return f"{rank_dict[hand[0]]} of {suit_sample[0]} and {rank_dict[hand[1]]} of {suit_sample[1]}"
    #     else:
    #         if 's' in hand:
    #             return f"{rank_dict[hand[0]]} of {suit_sample[0]} and {rank_dict[hand[1]]} of {suit_sample[0]}"
    #         else:
    #             return f"{rank_dict[hand[0]]} of {suit_sample[0]} and {rank_dict[hand[1]]} of {suit_sample[1]}"
            
    def parse_holding(holding):
        rank_map = {"2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine", 
                    "T": "Ten", "J": "Jack", "Q": "Queen", "K": "King", "A": "Ace"}
        suit_map = {'h': 'Heart', 'c': 'Club', 'd': 'Diamond', 's': 'Spade'}

        return f"[{rank_map[holding[0]]} of {suit_map[holding[1]]} and {rank_map[holding[2]]} of {suit_map[holding[3]]}]"
    def parse_moves(moves):
        moves_ls = ast.literal_eval(moves)
        return [f"raise {move.split('bb')[0]}" if 'bb' in move else move.upper() for move in moves_ls]
    def construct_prompt(row: pd.Series, tokenizer):
        # print(row)
        preflop_action_summary = parse_action(row['prev_line'])
        hero_position = row['hero_pos']
        hero_holding = parse_holding(row['hero_holding'])
        current_pot_size = row['pot_size']
        available_moves = parse_moves(row['available_moves'])
        best_current_hand = sergio_custom_function() # placeholder code, Sergio, please modify this line after you finished implementing your function.

        # Build the prompt using segments with dynamic indicators
        segments = []
        segments.append(("You are a specialist in playing 6-handed No Limit Texas Holdem. The following will be a game scenario and you need to make the optimal decision.\n\nHere is a game summary:\n\n", 0))
        segments.append(("The small blind is 0.5 chips and the big blind is 1 chips. Everyone started with 100 chips.\nThe player positions involved in this game are UTG, HJ, CO, BTN, SB, BB.\n", 0))
        segments.append(("In this hand, your position is ", 0))
        segments.append((hero_position, 1))
        segments.append((", and your holding is ", 0))
        segments.append((hero_holding, 1))
        segments.append((".\n", 0))
        segments.append(("You currently have ", 0))
        segments.append((best_current_hand, 1))
        segments.append((".\n", 0))
        segments.append(("Before the flop, ", 0))
        segments.append((preflop_action_summary, 1))
        segments.append((". Assume that all other players that is not mentioned folded.\n\nNow it is your turn to make a move.\nTo remind you, the current pot size is ", 0))
        segments.append((str(current_pot_size), 1))
        segments.append((" chips, and your holding is ", 0))
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

        correct_decision = f"raise {row['correct_decision'].split('bb')[0]}" if 'bb' in row['correct_decision'] else row['correct_decision'].lower()

        return prompt_text, available_moves, correct_decision, binary_indicator
    
    model_name = "meta-llama/meta-llama-3.1-70b-instruct" # can move this to function parameters if needed
    tokenizer = initialize_tokenizer(model_name=model_name)
    preflop_dataset_json = [
        {
            "instruction": construct_prompt(preflop_dataset.iloc[i], tokenizer)[0],
            "output": construct_prompt(preflop_dataset.iloc[i], tokenizer)[2],
            "binary_indicator": construct_prompt(preflop_dataset.iloc[i], tokenizer)[3]
        } 
        for i in range(preflop_dataset.shape[0])
    ]
    return preflop_dataset_json

def postflop_csv_to_json(postflop_dataset: pd.DataFrame):
    def parse_preflop_action(preflop_action):
        position_map = {"UTG": "UTG", "HJ": "HJ", "CO": "CO", 
                        "BTN": "BTN", "SB": "SB", "BB": "BB"}
        action_list = preflop_action.split('/')
        if len(action_list) == 4:
            position_1 = position_map[action_list[0]]
            position_1_raise_size = action_list[1]
            position_2 = position_map[action_list[2]]
            position_2_action = action_list[3]
            return f"{position_1} RAISE {position_1_raise_size}, and {position_2} CALL"
        elif len(action_list) == 6:
            position_1 = position_map[action_list[0]]
            position_1_raise_size = action_list[1]
            position_2 = position_map[action_list[2]]
            position_2_reraise_size = action_list[3]
            position_1_action = action_list[5]
            return f"{position_1} RAISE {position_1_raise_size}, {position_2} RAISE {position_2_reraise_size}, and {position_1} CALL"
        else:
            raise ValueError("Unseen Preflop Action")
    
    def parse_board(board):
        rank_map = {"2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine", 
                    "T": "Ten", "J": "Jack", "Q": "Queen", "K": "King", "A": "Ace"}
        suit_map = {'h': 'Heart', 'c': 'Club', 'd': 'Diamond', 's': 'Spade'}
        if len(board) > 2:
            board_list = [board[0:2], board[2:4], board[4:6]]
        else:
            board_list = [board]
        processed_board_list = [f"{rank_map[card[0]]} Of {suit_map[card[1]]}" for card in board_list]
        return ', '.join(processed_board_list[:-1]) + ', and ' + processed_board_list[-1] \
            if len(processed_board_list) > 1 else processed_board_list[0]
    
    def parse_relative_position(preflop_action):
        position_map = {"UTG": "UTG", "HJ": "HJ", "CO": "CO", 
                        "BTN": "BTN", "SB": "SB", "BB": "BB"}
        relative_position_map = {"UTG": 3, "HJ": 4, "CO": 5, "BTN": 6, "SB": 1, "BB": 2}
        action_list = preflop_action.split('/')
        position_1 = action_list[0]
        position_2 = action_list[2]
        return {'OOP': position_map[position_2], 'IP': position_map[position_1]} if relative_position_map[position_1] > relative_position_map[position_2] else {
            'OOP': position_map[position_1], 'IP': position_map[position_2]}
    
    def parse_postflop_action(preflop_action, postflop_action):
        # relative_position_map = parse_relative_position(preflop_action)  # Not needed if actions include player positions
        if pd.isna(postflop_action) or postflop_action == "":
            return {"flop": "there has been no action yet"}
        action_list = postflop_action.split('/')
        
        def process_action_list(action_list):
            processed_action_list = []
            for action in action_list:
                if "CHECK" in action.upper():
                    action = action.replace("_", " ").upper()
                if "BET" in action.upper() or "RAISE" in action.upper():
                    action = action.replace("_", " ").upper() + " chips"
                elif 'CALL' in action.upper():
                    action = action.replace("_", " ").upper()
                processed_action_list.append(action)
            if len(processed_action_list) == 0:
                return "there has been no action yet"
            return ', '.join(processed_action_list[:-1]) + ', and ' + processed_action_list[-1] \
                if len(processed_action_list) > 1 else processed_action_list[0]
        
        dealcards_indices = [i for i, action in enumerate(action_list) if action == 'dealcards']
        if len(dealcards_indices) == 1:
            sep_index = dealcards_indices[0]
            flop_action_list = action_list[:sep_index]
            turn_action_list = action_list[sep_index+2:]
            processed_flop_action_list = process_action_list(flop_action_list)
            processed_turn_action_list = process_action_list(turn_action_list)
            return {"flop": processed_flop_action_list, "turn": processed_turn_action_list}
        elif len(dealcards_indices) == 2:
            sep_index_1 = dealcards_indices[0]
            sep_index_2 = dealcards_indices[1]
            flop_action_list = action_list[:sep_index_1]
            turn_action_list = action_list[sep_index_1 + 2:sep_index_2]
            river_action_list = action_list[sep_index_2 + 2:]
            processed_flop_action_list = process_action_list(flop_action_list)
            processed_turn_action_list = process_action_list(turn_action_list)
            processed_river_action_list = process_action_list(river_action_list)
            return {"flop": processed_flop_action_list, "turn": processed_turn_action_list, "river": processed_river_action_list}
        else:
            processed_flop_action_list = process_action_list(action_list)
            return {"flop": processed_flop_action_list}
    
    def parse_holding(holding):
        rank_map = {"2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine", 
                    "T": "Ten", "J": "Jack", "Q": "Queen", "K": "King", "A": "Ace"}
        suit_map = {'h': 'Heart', 'c': 'Club', 'd': 'Diamond', 's': 'Spade'}
        return f"[{rank_map[holding[0]]} of {suit_map[holding[1]]} and {rank_map[holding[2]]} of {suit_map[holding[3]]}]"
    
    def parse_available_moves(available_moves):
        def parse_bet_raise(move):
            action_name, amount = move.rsplit(' ', 1)
            return f"{action_name} {int(round(float(amount)))}"
        return ", ".join([parse_bet_raise(move) if ("BET" in move.upper() or "RAISE" in move.upper()) else move.lower() for move in available_moves])
    
    def construct_prompt(row: pd.Seriesm, tokenizer):
        # relative_position_map = parse_relative_position(row['preflop_action'])  # Not needed
        # hero_position = relative_position_map[row['hero_position']]  # Directly use hero_position
        hero_position = row['hero_position']
        hero_holding = parse_holding(row['holding'])
        preflop_action_summary = parse_preflop_action(row['preflop_action']).replace("bb", " chips")
        postflop_actions = parse_postflop_action(row['preflop_action'], row['postflop_action'])
        flop_summary = f"The flop comes {parse_board(row['board_flop'])}, then {postflop_actions.get('flop', '')}."
        eval_at_turn = row['evaluation_at'] == "Turn" or row['evaluation_at'] == "River"
        eval_at_river = row['evaluation_at'] == "River"
        if eval_at_turn and 'turn' in postflop_actions:
            turn_summary = f"The turn comes {parse_board(row['board_turn'])}, then {postflop_actions['turn']}."
        else:
            turn_summary = ""
        if eval_at_river and 'river' in postflop_actions:
            river_summary = f"The river comes {parse_board(row['board_river'])}, then {postflop_actions['river']}."
        else:
            river_summary = ""
        current_pot_size = row['pot_size']
        available_moves = parse_available_moves(ast.literal_eval(row['available_moves']))
        best_current_hand = sergio_custom_function() # placeholder code, Sergio, please modify this line after you finished implementing your function.

        # Build the prompt using segments with dynamic indicators
        segments = []
        segments.append(("You are a specialist in playing 6-handed No Limit Texas Holdem. The following will be a game scenario and you need to make the optimal decision.\n\nHere is a game summary:\n\n", 0))
        segments.append(("The small blind is 0.5 chips and the big blind is 1 chips. Everyone started with 100 chips.\nThe player positions involved in this game are UTG, HJ, CO, BTN, SB, BB.\n", 0))
        segments.append(("In this hand, your position is ", 0))
        segments.append((hero_position, 1))
        segments.append((", and your holding is ", 0))
        segments.append((hero_holding, 1))
        segments.append((".\nBefore the flop, ", 0))
        segments.append((preflop_action_summary, 1))
        segments.append((". Assume that all other players that is not mentioned folded.\n", 0))
        if flop_summary:
            segments.append((flop_summary + "\n", 1))
        if turn_summary:
            segments.append((turn_summary + "\n", 1))
        if river_summary:
            segments.append((river_summary + "\n", 1))
        segments.append(("You currently have ", 0))
        segments.append((best_current_hand, 1))
        segments.append((".\n\nNow it is your turn to make a move.\nTo remind you, the current pot size is ", 0))
        segments.append((str(current_pot_size), 1))
        segments.append((" chips, and your holding is ", 0))
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
        
        correct_decision = row['correct_decision'].lower()

        return prompt_text, correct_decision, binary_indicator
    
    model_name = "meta-llama/meta-llama-3.1-70b-instruct" # can move this to function parameters if needed
    tokenizer = initialize_tokenizer(model_name=model_name)
    postflop_dataset_json = [
        {
            "instruction": construct_prompt(postflop_dataset.iloc[i], tokenizer)[0],
            "output": construct_prompt(postflop_dataset.iloc[i], tokenizer)[1],
            "binary_indicator": construct_prompt(postflop_dataset.iloc[i], tokenizer)[2]
        } 
        for i in range(postflop_dataset.shape[0])
    ]
    return postflop_dataset_json

def poker_csv_to_json(dataset: pd.DataFrame, preflop=True):
    def replace_keywords(data):
        if isinstance(data, dict):
            return {k: replace_keywords(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [replace_keywords(item) for item in data]
        elif isinstance(data, str):
            data = data.replace("ALLIN", "all in")
            data = data.replace("AllIn", "all in")
            data = data.replace("CALL", "call")
            data = data.replace("RAISE", "raise")
            data = data.replace("CHECK", "check")
            data = data.replace("BET", "bet")
            data = data.replace("FOLD", "fold")
            data = data.replace("UNDER_THE_GUN", "UTG")
            data = data.replace("HIJACK", "HJ")
            data = data.replace("CUTOFF", "CO")
            data = data.replace("BUTTON", "BTN")
            data = data.replace("SMALL_BLIND", "SB")
            data = data.replace("BIG_BLIND", "BB")
            data = data.replace("BIG_BLIND", "BB")
            return data
        else:
            return data
        
    if preflop:
        dataset_json = preflop_csv_to_json(dataset)
    else:
        dataset_json = postflop_csv_to_json(dataset)
    
    dataset_json = replace_keywords(dataset_json)

    return dataset_json

if __name__ == "__main__":
    CSV_FILENAME = "preflop_1k_test_set.csv"
    IS_PREFLOP = False
    JSON_FILENAME = "sergio_validation_preflop_1k_test_set.json"

    dataset = pd.read_csv(CSV_FILENAME).fillna("")
    dataset_json = poker_csv_to_json(dataset, preflop=IS_PREFLOP)
    with open(JSON_FILENAME, 'w') as json_file:
        random.shuffle(dataset_json)
        json.dump(dataset_json, json_file, indent=2)

# implementation todos:
# 1. replace placeholder function (probably just import from another file also fine)
# 2. make sure you change the placeholder function in both the preflop and postflop processing functions (sergio_custom_function())

# validation todos:
# 1. make sure preflop works
# 2. make sure postflop works