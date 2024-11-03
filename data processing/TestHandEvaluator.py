import unittest
from HandEvaluator import PokerHandEvaluator  

def test_poker_hand_evaluator():
    # Define test scenarios with the specified board input format
    test_cases = [
        {
            "scenario": "Preflop - High Card",
            "hero_holding": ['As', 'Kd'],
            "board": [],  
            "expected": ("High Card", "Ace-high")
        },
        {
            "scenario": "Preflop - One Pair",
            "hero_holding": ['Qh', 'Qc'],
            "board": [], 
            "expected": ("One Pair", "One Pair, Queens")
        },
        {
            "scenario": "Postflop - Two Pair",
            "hero_holding": ['8h', '8c'],
            "board": ['Ks7h2d', 'Jc', '7c'],  
            "expected": ("Two Pair", "Two Pair, Eights and Sevens with King kicker")
        },
        {
            "scenario": "Postflop - Straight",
            "hero_holding": ['6h', '5s'],
            "board": ['7h4d3c', '2s', '8c'],
            "expected": ("Straight", "Straight, Four to Eight")
        },
        {
            "scenario": "Postflop - Flush",
            "hero_holding": ['Ah', '2h'],
            "board": ['Kh7h4h', '3h', '8c'],
            "expected": ("Flush", "Flush, Ace-high")
        },
        {
            "scenario": "Postflop - Full House",
            "hero_holding": ['Qs', 'Qd'],
            "board": ['Qh7c7s', '5d', '2c'],
            "expected": ("Full House", "Full House, Queens over Sevens")
        },
        {
            "scenario": "Postflop - Four of a Kind",
            "hero_holding": ['Jd', 'Jh'],
            "board": ['JsJc2d', '8h', '3c'],
            "expected": ("Four of a Kind", "Four of a Kind, Jacks with 8 kicker")
        },
        {
            "scenario": "Postflop - Royal Flush",
            "hero_holding": ['As', 'Ks'],
            "board": ['QsJs10s', '8d', '3h'],
            "expected": ("Royal Flush", "Ace-high Royal Flush")
        }
    ]

    for case in test_cases:
        # Initialize evaluator based on preflop or postflop scenario
        evaluator = PokerHandEvaluator(case["hero_holding"], case["board"])
        
        best_hand, description = evaluator.get_best_hand()
        print(f"Scenario: {case['scenario']}")
        print(f"Expected: {case['expected']}")
        print(f"Result: ({best_hand}, {description})\n")
        assert (best_hand, description) == case["expected"], f"Failed {case['scenario']}"

# Run the test
test_poker_hand_evaluator()