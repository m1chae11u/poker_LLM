# poker_range_parser.py

import itertools
import re

def generate_poker_hands(range_str):
    CARD_RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    RANK_TO_INDEX = {rank: index for index, rank in enumerate(CARD_RANKS)}
    SUITS = ['h', 'd', 's', 'c']
    hands = []

    ranges = range_str.replace(" ", "").split(',')

    for hand_range in ranges:
        # Match pairs
        match = re.match(r'^([2-9TJQKA])\1(\+)?$', hand_range)
        if match:
            rank, plus = match.groups()
            start_index = RANK_TO_INDEX[rank]
            indices = range(start_index, len(CARD_RANKS)) if plus else [start_index]
            for index in indices:
                rank = CARD_RANKS[index]
                # Generate all combinations of suits for pairs
                for suit1, suit2 in itertools.combinations(SUITS, 2):
                    hands.append([rank + suit1, rank + suit2])
            continue

        # Match suited hands
        match = re.match(r'^([2-9TJQKA])([2-9TJQKA])s(\+)?$', hand_range)
        if match:
            first_rank, second_rank, plus = match.groups()
            first_index = RANK_TO_INDEX[first_rank]
            second_index = RANK_TO_INDEX[second_rank]
            indices = range(second_index, len(CARD_RANKS)) if plus else [second_index]
            for index in indices:
                second_rank = CARD_RANKS[index]
                # Generate suited combinations
                for suit in SUITS:
                    hands.append([first_rank + suit, second_rank + suit])
            continue

        # Match offsuit hands
        match = re.match(r'^([2-9TJQKA])([2-9TJQKA])o(\+)?$', hand_range)
        if match:
            first_rank, second_rank, plus = match.groups()
            first_index = RANK_TO_INDEX[first_rank]
            second_index = RANK_TO_INDEX[second_rank]
            indices = range(second_index, len(CARD_RANKS)) if plus else [second_index]
            for index in indices:
                second_rank = CARD_RANKS[index]
                # Generate offsuit combinations
                for suit1 in SUITS:
                    for suit2 in SUITS:
                        if suit1 != suit2:
                            hands.append([first_rank + suit1, second_rank + suit2])
            continue

        # Match both suited and offsuit hands (e.g., "KQ")
        match = re.match(r'^([2-9TJQKA])([2-9TJQKA])(\+)?$', hand_range)
        if match:
            first_rank, second_rank, plus = match.groups()
            first_index = RANK_TO_INDEX[first_rank]
            second_index = RANK_TO_INDEX[second_rank]
            indices = range(second_index, len(CARD_RANKS)) if plus else [second_index]
            for index in indices:
                second_rank = CARD_RANKS[index]
                # Generate both suited and offsuit combinations
                for suit1 in SUITS:
                    for suit2 in SUITS:
                        if suit1 == suit2:
                            hands.append([first_rank + suit1, second_rank + suit2])  # Suited
                        else:
                            hands.append([first_rank + suit1, second_rank + suit2])  # Offsuit
            continue

    return hands

if __name__ == "__main__":
    range_str = "66+, A8s+, KTs+, QTs+, AJo+, KQo"
    result = generate_poker_hands(range_str)
    print(result)