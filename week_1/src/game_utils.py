import numpy as np

def show_game_state(game_state, user_input=False):
    import pandas as pd
    from tabulate import tabulate
    assert len(game_state) == 9, "Game state must have exactly 9 elements."
    # print(f"\t | 1 \t| 2 \t| 3")
    # print(f"---------------------------------")
    # print(f"A \t| {game_state[0]} \t| {game_state[1]} \t| {game_state[2]}")
    # print(f"---------------------------------")
    # print(f"B \t| {game_state[3]} \t| {game_state[4]} \t| {game_state[5]}")
    # print(f"---------------------------------")
    # print(f"C \t| {game_state[6]} \t| {game_state[7]} \t| {game_state[8]}")
    # print(f"---------------------------------")
    if user_input:
        print("Current game state (you are 'X'):")
    game_state_2d = game_state.reshape(3, 3)
    symbols = {1: 'X', -1: 'O', 0: ' '}
    # show in neat board format
    game_df = pd.DataFrame(game_state_2d)
    game_df = game_df.replace(symbols)
    # If user input show rows as A, B, C and columns as 1, 2, 3

    if user_input:
        game_df.index = ['A', 'B', 'C']
        game_df.columns = ['1', '2', '3']
        print(tabulate(game_df, tablefmt='grid', showindex=True, headers=game_df.columns))
    else:
        print(tabulate(game_df, tablefmt='grid', showindex=False))

def show_game_sequence(game_sequence, persist_final_state=True):
    from IPython.display import clear_output
    from time import sleep
    for i, game_state in enumerate(game_sequence["all_game_states"][:-1]):
        print(f"Step {i+1}:")
        show_game_state(game_state)
        sleep(1)
        clear_output(wait=True)
    if persist_final_state:
        print(f"Final Step:")
        show_game_state(game_sequence["all_game_states"][-1])
        
        
def is_game_over(game_state):
    assert len(game_state) == 9, "Game state must have exactly 9 elements."
    game_state_2d = game_state.reshape(3, 3)
    # numbers = +1 or -1 for players, 0 for empty

    row_sums = game_state_2d.sum(axis=1)
    col_sums = game_state_2d.sum(axis=0)
    diag1_sum = game_state_2d.trace()
    game_state_flipped = np.fliplr(game_state_2d)
    diag2_sum = game_state_flipped.trace()
    winning_sums = [3, -3]
    all_sums = list(row_sums) + list(col_sums) + [diag1_sum, diag2_sum]
    for s in all_sums:
        if s in winning_sums:
            return True, s
    
    unique, counts = np.unique(game_state, return_counts=True)
    if 0 not in unique:
        return True, 0

    return False, 0

def run_random_game(verbose=False, return_game_states=False):
    game_sequence = {}
    game_sequence["all_game_states"] = []
    game_state = np.zeros(9, dtype=int)
    current_player = np.random.choice([1, -1]) # +1 for player 1, -1 for player 2
    first_player = current_player.copy()
    opponent = -current_player
    game_over = False
    all_user_inputs = [(game_state.copy(), 0, -1)]  # Initial state with no move
    while not game_over:
        available_moves = np.where(game_state == 0)[0]
        chosen_move = np.random.choice(available_moves)
        all_user_inputs.append((game_state.copy(), current_player, chosen_move))
        game_state[chosen_move] = current_player
        game_sequence["all_game_states"].append(game_state.copy())
        game_over, result = is_game_over(game_state)
        if game_over:
            if verbose:
                show_game_state(game_state)
            if result == 3:
                if verbose:
                    print("Player 1 wins!")
            elif result == -3:
                if verbose:
                    print("Player 2 wins!")
            else:
                if verbose:
                    print("It's a draw!")
            who_won = 1 if result == 3 else (-1 if result == -3 else 0)
            first_player_won = True if who_won == first_player else False
            game_sequence["who_won"] = "Player 1 [X]" if who_won == 1 else ("Player 2 [0]" if who_won == -1 else "Draw :(")
            game_sequence["all_game_states"].append(game_state.copy())  # Append final state    
            return game_sequence
        current_player, opponent = opponent, current_player
        

def user_plays_against_computer(verbose=False, return_game_states=False):
    from IPython.display import clear_output
    game_sequence = {}
    game_sequence["all_game_states"] = []
    game_state = np.zeros(9, dtype=int)
    current_player = np.random.choice([1, -1]) # +1 for user, -1 for computer
    first_player = current_player.copy()
    opponent = -current_player
    game_over = False
    all_user_inputs = [(game_state.copy(), 0, -1)]  # Initial state with no move
    while not game_over:
        available_moves = np.where(game_state == 0)[0]
        if current_player == -1:
            chosen_move = np.random.choice(available_moves)
        else:
            # Show current game state
            show_game_state(game_state, user_input=True)
            chosen_move_input = input(f"Enter your move (A1, B1, C3...) ")
            # map chosen_move_input to index
            mapping = {"A1": 0, "A2": 1, "A3": 2,
                          "B1": 3, "B2": 4, "B3": 5,
                          "C1": 6, "C2": 7, "C3": 8}
            chosen_move = mapping.get(chosen_move_input.upper(), -1)

            clear_output(wait=True)
            while chosen_move not in available_moves:
                print(f"Invalid move. Available moves: {available_moves}")
                chosen_move = int(input(f"Enter your move (0-8): "))
        all_user_inputs.append((game_state.copy(), current_player, chosen_move))
        game_state[chosen_move] = current_player
        game_sequence["all_game_states"].append(game_state.copy())
        game_over, result = is_game_over(game_state)
        if game_over:
            if verbose:
                show_game_state(game_state, user_input=True)
            if result == 3:
                if verbose:
                    print("You win! :) ")
            elif result == -3:
                if verbose:
                    print("Computer wins! :( ")
            else:
                if verbose:
                    print("It's a draw! :/ ")
            who_won = 1 if result == 3 else (-1 if result == -3 else 0)
            first_player_won = True if who_won == first_player else False
            game_sequence["who_won"] = "Player 1 [X]" if who_won == 1 else ("Player 2 [0]" if who_won == -1 else "Draw :(")
            game_sequence["all_game_states"].append(game_state.copy())  # Append final state    
            return game_sequence
        current_player, opponent = opponent, current_player
        

def minimax_tictactoe(game_state):
    """
    Minimax algorithm for tic-tac-toe.
    
    Args:
        game_state: numpy array of size 9
                   +1 = user (maximizer)
                   -1 = AI/minimax (minimizer)
                    0 = empty
    
    Returns:
        best_move: index (0-8) of the best move for the current player
    """
    
    def check_winner(state):
        """Check if there's a winner. Returns +1, -1, or 0."""
        # Reshape to 3x3 for easier checking
        board = state.reshape(3, 3)
        
        # Check rows
        for row in board:
            if abs(sum(row)) == 3:
                return row[0]
        
        # Check columns
        for col in range(3):
            col_sum = sum(board[:, col])
            if abs(col_sum) == 3:
                return board[0, col]
        
        # Check diagonals
        diag1 = board[0, 0] + board[1, 1] + board[2, 2]
        if abs(diag1) == 3:
            return board[1, 1]
        
        diag2 = board[0, 2] + board[1, 1] + board[2, 0]
        if abs(diag2) == 3:
            return board[1, 1]
        
        return 0
    
    def is_terminal(state):
        """Check if game is over."""
        return check_winner(state) != 0 or 0 not in state
    
    def minimax(state, is_maximizing):
        """Recursive minimax function."""
        winner = check_winner(state)
        
        # Terminal states
        if winner == 1:
            return 10  # User wins
        elif winner == -1:
            return -10  # AI wins
        elif 0 not in state:
            return 0  # Draw
        
        if is_maximizing:
            # User's turn (maximizer)
            max_eval = -np.inf
            for i in range(9):
                if state[i] == 0:
                    state[i] = 1
                    eval_score = minimax(state, False)
                    state[i] = 0
                    max_eval = max(max_eval, eval_score)
            return max_eval
        else:
            # AI's turn (minimizer)
            min_eval = np.inf
            for i in range(9):
                if state[i] == 0:
                    state[i] = -1
                    eval_score = minimax(state, True)
                    state[i] = 0
                    min_eval = min(min_eval, eval_score)
            return min_eval
    
    # Determine whose turn it is based on piece count
    count_user = np.sum(game_state == 1)
    count_ai = np.sum(game_state == -1)
    is_ai_turn = count_user > count_ai
    
    # Find best move
    best_move = -1
    best_value = np.inf if is_ai_turn else -np.inf
    
    for i in range(9):
        if game_state[i] == 0:
            # Try this move
            game_state[i] = -1 if is_ai_turn else 1
            move_value = minimax(game_state, not is_ai_turn)
            game_state[i] = 0
            
            # Update best move
            if is_ai_turn:
                if move_value < best_value:
                    best_value = move_value
                    best_move = i
            else:
                if move_value > best_value:
                    best_value = move_value
                    best_move = i
    
    return best_move

def minimax_lookup(game_state):
    """
    Precomputed minimax optimal moves.
    This is the actual minimax solution encoded as rules.
    """
    
    def check_win_or_block(state, player):
        lines = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
        for line in lines:
            vals = [state[i] for i in line]
            if vals.count(player) == 2 and vals.count(0) == 1:
                return line[vals.index(0)]
        return None
    
    # Win or block
    for player in [-1, 1]:
        move = check_win_or_block(game_state, player)
        if move is not None:
            return move
    
    # Optimal opening moves (first move)
    if np.sum(game_state != 0) == 0:
        return 0  # Corner
    
    # Response to first move
    if np.sum(game_state != 0) == 1:
        if game_state[4] == 1:  # Opponent took center
            return 0  # Take corner
        else:
            return 4  # Take center
    
    # Center if available
    if game_state[4] == 0:
        return 4
    
    # Corners, prioritizing opposite corner
    corners = [0, 2, 6, 8]
    opp_corner = {0:8, 8:0, 2:6, 6:2}
    for c in corners:
        if game_state[c] == 1 and game_state[opp_corner[c]] == 0:
            return opp_corner[c]
    
    for c in corners:
        if game_state[c] == 0:
            return c
    
    # Edges
    for e in [1, 3, 5, 7]:
        if game_state[e] == 0:
            return e
    
    return None

def run_random_game_with_minimax(verbose=False):
    # minimax = -1 for AI, +1 for random computer player
    game_state = np.zeros(9, dtype=int)
    current_player = np.random.choice([-1, 1])
    first_player = current_player.copy()
    opponent = -current_player
    game_over = False
    all_user_inputs = [(game_state.copy(), 0, -1)]  # Initial state with no move
    while not game_over:
        available_moves = np.where(game_state == 0)[0]
        if current_player == -1:
            chosen_move = minimax_lookup(game_state)
        else:
            chosen_move = np.random.choice(available_moves)
        all_user_inputs.append((game_state.copy(), current_player, chosen_move))
        game_state[chosen_move] = current_player
        game_over, result = is_game_over(game_state)
        if game_over:
            if verbose:
                show_game_state(game_state)
            if result == 3:
                if verbose:
                    print("Player 1 wins!")
            elif result == -3:
                if verbose:
                    print("Player 2 wins!")
            else:
                if verbose:
                    print("It's a draw!")
            who_won = 1 if result == 3 else (-1 if result == -3 else 0)
            first_player_won = True if who_won == first_player else False
            return who_won, all_user_inputs, first_player_won
        
        current_player, opponent = opponent, current_player

def model_predict_next_move(game_state, model):
    import torch
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(game_state, dtype=torch.float32).unsqueeze(0)
        output = model(input_tensor)
        predicted_move = torch.argmax(output, dim=1).item()
    return predicted_move

def run_random_game_with_model(model_input, verbose=False):
    # -1 for model, +1 for random computer player
    game_state = np.zeros(9, dtype=int)
    current_player = np.random.choice([-1, 1])
    first_player = current_player.copy()
    opponent = -current_player
    game_over = False
    all_user_inputs = [(game_state.copy(), 0, -1)]  # Initial state with no move
    while not game_over:
        available_moves = np.where(game_state == 0)[0]
        if current_player == -1:
            chosen_move = model_predict_next_move(game_state, model_input)
        else:
            chosen_move = np.random.choice(available_moves)
        all_user_inputs.append((game_state.copy(), current_player, chosen_move))
        game_state[chosen_move] = current_player
        game_over, result = is_game_over(game_state)
        if game_over:
            if verbose:
                show_game_state(game_state)
            if result == 3:
                if verbose:
                    print("Player 1 wins!")
            elif result == -3:
                if verbose:
                    print("Player 2 wins!")
            else:
                if verbose:
                    print("It's a draw!")
            who_won = 1 if result == 3 else (-1 if result == -3 else 0)
            first_player_won = True if who_won == first_player else False
            return who_won, all_user_inputs, first_player_won
        
        current_player, opponent = opponent, current_player

def run_models_against_each_other(model_1, model_2, verbose=False):
    # -1 for model, +1 for random computer player
    game_state = np.zeros(9, dtype=int)
    current_player = np.random.choice([-1, 1])
    first_player = current_player.copy()
    opponent = -current_player
    game_over = False
    all_user_inputs = [(game_state.copy(), 0, -1)]  # Initial state with no move
    while not game_over:
        available_moves = np.where(game_state == 0)[0]
        if current_player == -1:
            chosen_move = model_predict_next_move(game_state, model_2)
        else:
            chosen_move = np.random.choice(available_moves)
        all_user_inputs.append((game_state.copy(), current_player, chosen_move))
        game_state[chosen_move] = current_player
        game_over, result = is_game_over(game_state)
        if game_over:
            if verbose:
                show_game_state(game_state)
            if result == 3:
                if verbose:
                    print("Player 1 wins!")
            elif result == -3:
                if verbose:
                    print("Player 2 wins!")
            else:
                if verbose:
                    print("It's a draw!")
            who_won = 1 if result == 3 else (-1 if result == -3 else 0)
            first_player_won = True if who_won == first_player else False
            return who_won, all_user_inputs, first_player_won
        
        current_player, opponent = opponent, current_player

def run_models_against_minimax(model_input, verbose=False):
    # -1 for model, +1 for random computer player
    game_state = np.zeros(9, dtype=int)
    current_player = np.random.choice([-1, 1])
    first_player = current_player.copy()
    opponent = -current_player
    game_over = False
    all_user_inputs = [(game_state.copy(), 0, -1)]  # Initial state with no move
    while not game_over:
        available_moves = np.where(game_state == 0)[0]
        if current_player == -1:
            chosen_move = model_predict_next_move(game_state, model_input)
        else:
            chosen_move = minimax_lookup(game_state)
        all_user_inputs.append((game_state.copy(), current_player, chosen_move))
        game_state[chosen_move] = current_player
        game_over, result = is_game_over(game_state)
        if game_over:
            if verbose:
                show_game_state(game_state)
            if result == 3:
                if verbose:
                    print("Player 1 wins!")
            elif result == -3:
                if verbose:
                    print("Player 2 wins!")
            else:
                if verbose:
                    print("It's a draw!")
            who_won = 1 if result == 3 else (-1 if result == -3 else 0)
            first_player_won = True if who_won == first_player else False
            return who_won, all_user_inputs, first_player_won
        
        current_player, opponent = opponent, current_player

