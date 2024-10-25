import subprocess
import json

def calculate_exact_equity(hero_hand, villain_range, board):
    command = [
        'node',
        './equity_calculator_exact.js', 
        json.dumps(hero_hand),     
        json.dumps(villain_range),
        json.dumps(board)
    ]
    
    result = subprocess.run(command, capture_output=True, text=True)
    equity_result = json.loads(result.stdout)
    
    win_percentage = equity_result['win'] * 100
    tie_percentage = equity_result['tie'] * 100

    return win_percentage, tie_percentage

def calculate_approximate_equity(hero_hand, villain_range, num_simulations=100000):
    command = [
        'node',
        './equity_calculator_approximate.js', 
        json.dumps(hero_hand),                 
        json.dumps(villain_range),
        str(num_simulations)                   
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    equity_result = json.loads(result.stdout)

    win_percentage = equity_result['win'] * 100
    tie_percentage = equity_result['tie'] * 100

    return win_percentage, tie_percentage

if __name__ == "__main__":
    hero_hand = ['As', 'Ks']
    # TT+ (TT, JJ, QQ, KK, AA)
    villain1_range = [
        ['Th', 'Td'], ['Th', 'Ts'], ['Th', 'Tc'], ['Td', 'Ts'], ['Td', 'Tc'], ['Ts', 'Tc'],
        ['Jh', 'Jd'], ['Jh', 'Js'], ['Jh', 'Jc'], ['Jd', 'Js'], ['Jd', 'Jc'], ['Js', 'Jc'],
        ['Qh', 'Qd'], ['Qh', 'Qs'], ['Qh', 'Qc'], ['Qd', 'Qs'], ['Qd', 'Qc'], ['Qs', 'Qc'],
        ['Kh', 'Kd'], ['Kh', 'Ks'], ['Kh', 'Kc'], ['Kd', 'Ks'], ['Kd', 'Kc'], ['Ks', 'Kc'],
        ['Ah', 'Ad'], ['Ah', 'As'], ['Ah', 'Ac'], ['Ad', 'As'], ['Ad', 'Ac'], ['As', 'Ac'],
    ]
    v2 = [
        ['6h', '6d'], ['6h', '6s'], ['6h', '6c'], ['6d', '6s'], ['6d', '6c'], ['6s', '6c'],
        ['7h', '7d'], ['7h', '7s'], ['7h', '7c'], ['7d', '7s'], ['7d', '7c'], ['7s', '7c'],
        ['8h', '8d'], ['8h', '8s'], ['8h', '8c'], ['8d', '8s'], ['8d', '8c'], ['8s', '8c'],
        ['9h', '9d'], ['9h', '9s'], ['9h', '9c'], ['9d', '9s'], ['9d', '9c'], ['9s', '9c'], 
        ['Th', 'Td'], ['Th', 'Ts'], ['Th', 'Tc'], ['Td', 'Ts'], ['Td', 'Tc'], ['Ts', 'Tc'],
        ['Jh', 'Jd'], ['Jh', 'Js'], ['Jh', 'Jc'], ['Jd', 'Js'], ['Jd', 'Jc'], ['Js', 'Jc'],
        ['Qh', 'Qd'], ['Qh', 'Qs'], ['Qh', 'Qc'], ['Qd', 'Qs'], ['Qd', 'Qc'], ['Qs', 'Qc'],
        ['Kh', 'Kd'], ['Kh', 'Ks'], ['Kh', 'Kc'], ['Kd', 'Ks'], ['Kd', 'Kc'], ['Ks', 'Kc'],
        ['Ah', 'Ad'], ['Ah', 'As'], ['Ah', 'Ac'], ['Ad', 'As'], ['Ad', 'Ac'], ['As', 'Ac'],

        ['Ah', '8h'], ['As', '8s'], ['Ad', '8d'], ['Ac', '8c'],
        ['Ah', '9h'], ['As', '9s'], ['Ad', '9d'], ['Ac', '9c'], 
        ['Ah', 'Th'], ['As', 'Ts'], ['Ad', 'Td'], ['Ac', 'Tc'], 
        ['Ah', 'Jh'], ['As', 'Js'], ['Ad', 'Jd'], ['Ac', 'Jc'], 
        ['Ah', 'Qh'], ['As', 'Qs'], ['Ad', 'Qd'], ['Ac', 'Qc'], 
        ['Ah', 'Kh'], ['As', 'Ks'], ['Ad', 'Kd'], ['Ac', 'Kc'], 

        ['Kh', 'Th'], ['Ks', 'Ts'], ['Kd', 'Td'], ['Kc', 'Tc'],
        ['Kh', 'Jh'], ['Ks', 'Js'], ['Kd', 'Jd'], ['Kc', 'Jc'],
        ['Kh', 'Qh'], ['Ks', 'Qs'], ['Kd', 'Qd'], ['Kc', 'Qc'], 

        ['Qh', 'Th'], ['Qs', 'Ts'], ['Qd', 'Td'], ['Qc', 'Tc'], 
        ['Qh', 'Jh'], ['Qs', 'Js'], ['Qd', 'Jd'], ['Qc', 'Jc'], 

        ['Ah', 'Jd'], ['Ah', 'Js'], ['Ah', 'Jc'], ['Ad', 'Jh'], ['Ad', 'Js'], ['Ad', 'Jc'], ['As', 'Jh'], ['As', 'Jd'], ['As', 'Jc'], ['Ac', 'Jh'], ['Ac', 'Jd'], ['Ac', 'Js'], 
        ['Ah', 'Qd'], ['Ah', 'Qs'], ['Ah', 'Qc'], ['Ad', 'Qh'], ['Ad', 'Qs'], ['Ad', 'Qc'], ['As', 'Qh'], ['As', 'Qd'], ['As', 'Qc'], ['Ac', 'Qh'], ['Ac', 'Qd'], ['Ac', 'Qs'], 
        ['Ah', 'Kd'], ['Ah', 'Ks'], ['Ah', 'Kc'], ['Ad', 'Kh'], ['Ad', 'Ks'], ['Ad', 'Kc'], ['As', 'Kh'], ['As', 'Kd'], ['As', 'Kc'], ['Ac', 'Kh'], ['Ac', 'Kd'], ['Ac', 'Ks'], 

        ['Kh', 'Qd'], ['Kh', 'Qs'], ['Kh', 'Qc'], ['Kd', 'Qh'], ['Kd', 'Qs'], ['Kd', 'Qc'], ['Ks', 'Qh'], ['Ks', 'Qd'], ['Ks', 'Qc'], ['Kc', 'Qh'], ['Kc', 'Qd'], ['Kc', 'Qs']
    ]
    board = ['2d', '7h', 'Jc', 'Qs', 'Td']
    empty_board = []

    exact_equity_result = calculate_exact_equity(hero_hand, v2, board)
    print(f"exact equity pre-flop (win, tie): {exact_equity_result}")

    approx_equity_result = calculate_approximate_equity(hero_hand, villain1_range)
    print(f"approx equity pre-flop (win, tie): {approx_equity_result}")

    # 66+, A8s+, KTs+, QTs+, AJo+, KQo
    '''
        ['6h', '6d'], ['6h', '6s'], ['6h', '6c'], ['6d', '6s'], ['6d', '6c'], ['6s', '6c'],
        ['7h', '7d'], ['7h', '7s'], ['7h', '7c'], ['7d', '7s'], ['7d', '7c'], ['7s', '7c'],
        ['8h', '8d'], ['8h', '8s'], ['8h', '8c'], ['8d', '8s'], ['8d', '8c'], ['8s', '8c'],
        ['9h', '9d'], ['9h', '9s'], ['9h', '9c'], ['9d', '9s'], ['9d', '9c'], ['9s', '9c'], 
        ['Th', 'Td'], ['Th', 'Ts'], ['Th', 'Tc'], ['Td', 'Ts'], ['Td', 'Tc'], ['Ts', 'Tc'],
        ['Jh', 'Jd'], ['Jh', 'Js'], ['Jh', 'Jc'], ['Jd', 'Js'], ['Jd', 'Jc'], ['Js', 'Jc'],
        ['Qh', 'Qd'], ['Qh', 'Qs'], ['Qh', 'Qc'], ['Qd', 'Qs'], ['Qd', 'Qc'], ['Qs', 'Qc'],
        ['Kh', 'Kd'], ['Kh', 'Ks'], ['Kh', 'Kc'], ['Kd', 'Ks'], ['Kd', 'Kc'], ['Ks', 'Kc'],
        ['Ah', 'Ad'], ['Ah', 'As'], ['Ah', 'Ac'], ['Ad', 'As'], ['Ad', 'Ac'], ['As', 'Ac'],

        ['Ah', '8h'], ['As', '8s'], ['Ad', '8d'], ['Ac', '8c'],
        ['Ah', '9h'], ['As', '9s'], ['Ad', '9d'], ['Ac', '9c'], 
        ['Ah', 'Th'], ['As', 'Ts'], ['Ad', 'Td'], ['Ac', 'Tc'], 
        ['Ah', 'Jh'], ['As', 'Js'], ['Ad', 'Jd'], ['Ac', 'Jc'], 
        ['Ah', 'Qh'], ['As', 'Qs'], ['Ad', 'Qd'], ['Ac', 'Qc'], 
        ['Ah', 'Kh'], ['As', 'Ks'], ['Ad', 'Kd'], ['Ac', 'Kc'], 

        ['Kh', 'Th'], ['Ks', 'Ts'], ['Kd', 'Td'], ['Kc', 'Tc'],
        ['Kh', 'Jh'], ['Ks', 'Js'], ['Kd', 'Jd'], ['Kc', 'Jc'],
        ['Kh', 'Qh'], ['Ks', 'Qs'], ['Kd', 'Qd'], ['Kc', 'Qc'], 

        ['Qh', 'Th'], ['Qs', 'Ts'], ['Qd', 'Td'], ['Qc', 'Tc'], 
        ['Qh', 'Jh'], ['Qs', 'Js'], ['Qd', 'Jd'], ['Qc', 'Jc'], 

        ['Ah', 'Jd'], ['Ah', 'Js'], ['Ah', 'Jc'], ['Ad', 'Jh'], ['Ad', 'Js'], ['Ad', 'Jc'], ['As', 'Jh'], ['As', 'Jd'], ['As', 'Jc'], ['Ac', 'Jh'], ['Ac', 'Jd'], ['Ac', 'Js'], 
        ['Ah', 'Qd'], ['Ah', 'Qs'], ['Ah', 'Qc'], ['Ad', 'Qh'], ['Ad', 'Qs'], ['Ad', 'Qc'], ['As', 'Qh'], ['As', 'Qd'], ['As', 'Qc'], ['Ac', 'Qh'], ['Ac', 'Qd'], ['Ac', 'Qs'], 
        ['Ah', 'Kd'], ['Ah', 'Ks'], ['Ah', 'Kc'], ['Ad', 'Kh'], ['Ad', 'Ks'], ['Ad', 'Kc'], ['As', 'Kh'], ['As', 'Kd'], ['As', 'Kc'], ['Ac', 'Kh'], ['Ac', 'Kd'], ['Ac', 'Ks'], 

        ['Kh', 'Qd'], ['Kh', 'Qs'], ['Kh', 'Qc'], ['Kd', 'Qh'], ['Kd', 'Qs'], ['Kd', 'Qc'], ['Ks', 'Qh'], ['Ks', 'Qd'], ['Ks', 'Qc'], ['Kc', 'Qh'], ['Kc', 'Qd'], ['Kc', 'Qs'] 
    '''
    