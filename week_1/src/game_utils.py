import numpy as np

def show_game_state(game_state, user_input=False, title=None):
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
    elif title is not None:
        print(title)

    game_state_2d = game_state.reshape(3, 3)
    symbols = {1: 'X', -1: 'O', 0: ' '}
    # show in neat board format
    game_df = pd.DataFrame(game_state_2d)
    game_df = game_df.replace(symbols)
    # If user input show rows as A, B, C and columns as 1, 2, 3

    game_df.index = ['A', 'B', 'C']
    game_df.columns = ['1', '2', '3']
    print(tabulate(game_df, tablefmt='grid', showindex=True, headers=game_df.columns))
    
def show_game_sequence(game_sequence, persist_final_state=True, players=None):
    from IPython.display import clear_output
    from time import sleep
    for i, game_state in enumerate(game_sequence["all_game_states"][:-1]):
        if players is None:
            players = {1: "Player 1 [X]", -1: "Player 2 [0]"}
        if i == 0:
            title_text = f"{game_sequence['first_player']} played first:"
            sleep(1)
            show_game_state(game_state, title=title_text)
        else:
            title_text = f"{current_player} played:"
            sleep(1)
            show_game_state(game_state, title=title_text)
        if i < len(game_sequence["all_user_inputs"])-2:
            current_player = players[game_sequence["all_user_inputs"][i+2][1]]
        else:
            current_player = "Game Over"
            #print(f"Game Over: {game_sequence['who_won']} won!")
            sleep(1)
        clear_output(wait=True)
    if persist_final_state:
        if game_sequence["who_won"] == "Draw :(":
            title_text = f"Game Over: It's a draw! :/ "
        else:
            title_text = f"Game Over: {game_sequence['who_won']} won!"
        show_game_state(game_sequence["all_game_states"][-1], title=title_text)
        
        
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
    game_sequence["first_player"] = "Player 1 [X]" if first_player == 1 else "Player 2 [0]"
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
            game_sequence["all_user_inputs"] = all_user_inputs
            return game_sequence
        current_player, opponent = opponent, current_player
        

def user_plays_against_computer(verbose=False, return_game_states=False):
    from IPython.display import clear_output
    from time import sleep
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
            show_game_state(game_state, user_input=False, title="Thinking...")
            sleep(1)
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

def test_model_for_illegal_moves(model_1, verbose=False):
    game_sequence = {}
    game_sequence["all_game_states"] = []
    game_state = np.zeros(9, dtype=int)
    current_player = np.random.choice([1, -1]) # +1 for player 1, -1 for player 2
    first_player = current_player.copy()
    game_sequence["first_player"] = "Player 1 [X]" if first_player == 1 else "Player 2 [0]"
    illegal_moves = []
    game_sequence["illegal_moves"] = illegal_moves
    opponent = -current_player
    game_over = False
    all_user_inputs = [(game_state.copy(), 0, -1)]  # Initial state with no move
    while not game_over:
        available_moves = np.where(game_state == 0)[0]
        if current_player == 1:
            chosen_move = model_predict_next_move(game_state, model_1)
            if chosen_move not in available_moves:
                if verbose:
                    show_game_state(game_state, title="Current game state:")
                    mapping = {0: "A1", 1: "A2", 2: "A3",
                                 3: "B1", 4: "B2", 5: "B3",
                                 6: "C1", 7: "C2", 8: "C3"}
                    print(f"Model predicted invalid move {mapping[chosen_move]}.")
                game_sequence["illegal_moves"].append((game_state.copy(), current_player, chosen_move))
                game_sequence["all_user_inputs"] = all_user_inputs
                return game_sequence 

        else:
            # Second player is the computer
            chosen_move = np.random.choice(available_moves)

        all_user_inputs.append((game_state.copy(), current_player, chosen_move))
        game_state[chosen_move] = current_player
        game_sequence["all_game_states"].append(game_state.copy())
        game_over, result = is_game_over(game_state)
        if game_over:
            if verbose:
                print("All moves were legal. Final game state:")
                show_game_state(game_state)
            if result == 3:
                if verbose:
                    print("All Player 1 wins!")
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
            game_sequence["all_user_inputs"] = all_user_inputs
            game_sequence["illegal_moves"] = None
            return game_sequence # Return True to indicate all moves were legal, along with the game sequence for analysis
        current_player, opponent = opponent, current_player

def run_models_against_each_other(model_1, model_2, model_1_name, model_2_name,
                                verbose=False, return_game_states=False, \
                                force_legal=True, max_retries=100):
    from src.train_game_utils import load_model_from_path
    if type(model_1) == str:
        # check if file exists
        import os
        if os.path.exists(model_1):
            model_1 = load_model_from_path(model_1)
        else:
            raise ValueError(f"Model file {model_1} does not exist.")
    if type(model_2) == str:
        # check if file exists
        import os
        if os.path.exists(model_2):
            model_2 = load_model_from_path(model_2)
        else:
            raise ValueError(f"Model file {model_2} does not exist.")

    model_1_name = "Player 1 [X]" if model_1_name is None else model_1_name
    model_2_name = "Player 2 [0]" if model_2_name is None else model_2_name
     
    game_sequence = {}
    game_sequence["all_game_states"] = []
    game_sequence["players"] = {1: model_1_name, -1: model_2_name}
    game_state = np.zeros(9, dtype=int)
    current_player = np.random.choice([1, -1]) # +1 for player 1, -1 for player 2
    first_player = current_player.copy()
    game_sequence["first_player"] = model_1_name if first_player == 1 else model_2_name
    illegal_moves = []
    game_sequence["illegal_moves"] = illegal_moves
    opponent = -current_player
    game_over = False
    all_user_inputs = [(game_state.copy(), 0, -1)]  # Initial state with no move
    first_move = True
    while not game_over:
        available_moves = np.where(game_state == 0)[0]
        if current_player == 1:
            decision_made_by_player_1 = False
            if not force_legal:
                chosen_move = model_predict_next_move(game_state, model_1)
                game_sequence["illegal_moves"].append((game_state.copy(), current_player, chosen_move))
                decision_made_by_player_1 = True
            num_retries = 0
            if first_move:
                chosen_move = np.random.choice(available_moves)  # For the first move, allow any move to avoid bias
                decision_made_by_player_1 = True
                first_move = False

            while not decision_made_by_player_1:
                chosen_move = model_predict_next_move(game_state, model_1)
                decision_made_by_player_1 = chosen_move in available_moves
                if not decision_made_by_player_1:
                    print(f"Model 1 predicted invalid move {chosen_move}. Retrying...")
                
                num_retries += 1
                if num_retries > max_retries:
                    if verbose:
                        print(f"Model 1 failed to produce a legal move after {max_retries} retries. Choosing randomly from available moves.")
                    game_sequence["illegal_moves"].append((game_state.copy(), current_player, chosen_move))
                    chosen_move = np.random.choice(available_moves)
                    decision_made_by_player_1 = True
        else:
            decision_made_by_player_2 = False
            if not force_legal:
                chosen_move = model_predict_next_move(game_state, model_2)
                game_sequence["illegal_moves"].append((game_state.copy(), current_player, chosen_move))
                decision_made_by_player_2 = True

            if first_move:
                chosen_move = np.random.choice(available_moves)
                decision_made_by_player_2 = True
                first_move = False
            num_retries = 0
            while not decision_made_by_player_2:
                chosen_move = model_predict_next_move(game_state, model_2)
                decision_made_by_player_2 = chosen_move in available_moves
                if not decision_made_by_player_2:
                    print(f"Model 2 predicted invalid move {chosen_move}. Retrying...")
                
                num_retries += 1
                if num_retries > max_retries:
                    if verbose:
                        print(f"Model 2 failed to produce a legal move after {max_retries} retries. Choosing randomly from available moves.")
                    game_sequence["illegal_moves"].append((game_state.copy(), current_player, chosen_move))
                    chosen_move = np.random.choice(available_moves)
                    decision_made_by_player_2 = True 

        all_user_inputs.append((game_state.copy(), current_player, chosen_move))
        game_state[chosen_move] = current_player
        game_sequence["all_game_states"].append(game_state.copy())
        game_over, result = is_game_over(game_state)
        if game_over:
            if verbose:
                show_game_state(game_state)
            if result == 3:
                if verbose:
                    print(f"{model_1_name} wins!")
            elif result == -3:
                if verbose:
                    print(f"{model_2_name} wins!")
            else:
                if verbose:
                    print("It's a draw!")
            who_won = 1 if result == 3 else (-1 if result == -3 else 0)
            first_player_won = True if who_won == first_player else False
            game_sequence["who_won"] = model_1_name if who_won == 1 else (model_2_name if who_won == -1 else "Draw :(")
            game_sequence["all_game_states"].append(game_state.copy())  # Append final state    
            game_sequence["all_user_inputs"] = all_user_inputs
            return game_sequence
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

