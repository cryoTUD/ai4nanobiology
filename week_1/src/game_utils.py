import numpy as np
class TicTacToeGame:
    def __init__(self, player_1="computer", player_2="computer", player_1_name=None, player_2_name=None):
        self.game_state = np.zeros(9, dtype=int)
        self.player_1 = player_1
        self.player_2 = player_2
        self.player_1_name = player_1_name
        self.player_2_name = player_2_name
        self.update_players()
        self.players = {1: self.player_1_name, -1: self.player_2_name}
        self.game_sequence = {}
        self.chosen_move = -1
        self.force_legal = True 
        self.max_retries = 100
        self.first_move = True
        self.verbose = False 
    
    def update_players(self):
        import os
        from src.train_game_utils import load_model_from_path
        # Player 1 can be the user, model or random player
        if self.player_1 == "user":
            self.player_1_type = "user"
            if self.player_1_name is None:
                self.player_1_name = "Player 1 [X]"
            else:
                self.player_1_name += " [X]"
        elif self.player_1 == "computer":
            self.player_1_type = "computer"
            if self.player_1_name is None:
                self.player_1_name = "Computer 1 [X]"
            else:
                self.player_1_name += " [X]"
        elif "load_state_dict" in dir(self.player_1):
            self.player_1_model = self.player_1
            self.player_1_type = "model"
            if self.player_1_name is None:
                self.player_1_name = "Custom Model 1 [X]"
            else:
                self.player_1_name += " [X]"
        elif os.path.exists(self.player_1):
            self.player_1_model = load_model_from_path(self.player_1)
            self.player_1_type = "model"
            if self.player_1_name is None:
                self.player_1_name = os.path.basename(self.player_1) + " [X]"
            else:
                self.player_1_name += " [X]"
        else:
            raise ValueError(f"Invalid player 1 type: {self.player_1}. \
                            \nCan be 'user', 'computer' or a file path to a model.")
        
        # Player 2 can be the user, model or random player
        if self.player_2 == "user":
            self.player_2_type = "user"
            if self.player_2_name is None:
                self.player_2_name = "Player 2 [O]"
            else:
                self.player_2_name += " [O]"
        elif self.player_2 == "computer":
            self.player_2_type = "computer"
            if self.player_2_name is None:
                self.player_2_name = "Computer 2 [O]"
            else:
                self.player_2_name += " [O]"
        elif "load_state_dict" in dir(self.player_2):
            self.player_2_model = self.player_2
            self.player_2_type = "model"
            if self.player_2_name is None:
                self.player_2_name = "Custom Model 2[X]"
            else:
                self.player_2_name += " [X]"
        elif os.path.exists(self.player_2):
            self.player_2_model = load_model_from_path(self.player_2)
            self.player_2_type = "model"
            if self.player_2_name is None:
                self.player_2_name = os.path.basename(self.player_2) + " [O]"
            else:
                self.player_2_name += " [O]"
        else:
            raise ValueError(f"Invalid player 2 type: {self.player_2}. \
                            \nCan be 'user', 'computer' or a file path to a model.")
        
    def play(self, verbose=False, force_legal=True, max_retries=100, randomise_first_move=True, visualise=False):
        from IPython.display import clear_output
        from time import sleep
        current_player = np.random.choice([1, -1])
        first_player = current_player.copy()
        opponent = -current_player
        game_over = False
        all_user_inputs = [(self.game_state.copy(), 0, self.chosen_move)]
        self.game_sequence["first_player"] = self.player_1_name if first_player == 1 else self.player_2_name
        self.game_sequence["players"] = {1: self.player_1_name, -1: self.player_2_name}
        illegal_moves = []
        self.game_sequence["illegal_moves"] = illegal_moves
        self.game_sequence["all_game_states"] = []

        while not game_over:
            available_moves = np.where(self.game_state == 0)[0]
            if current_player == 1:
                self.player_move(available_moves, player=1)
            else:
                self.player_move(available_moves, player=-1)
            self.first_move = False
            all_user_inputs.append((self.game_state.copy(), current_player, self.chosen_move)) 
            self.game_sequence["all_game_states"].append(self.game_state.copy())
            self.game_sequence["all_user_inputs"] = all_user_inputs
            game_over, result = self.is_game_over()

            # Exchange players for next turn
            current_player, opponent = opponent, current_player

        return self.finalize_game_sequence(result, verbose, visualise=visualise)
    
    def player_move(self, available_moves, player):
        if player == 1:
            self.player_1_move(available_moves, self.force_legal, self.max_retries, self.first_move, self.verbose)
        else:
            self.player_2_move(available_moves, self.force_legal, self.max_retries, self.first_move, self.verbose)

    def player_1_move(self, available_moves, force_legal, max_retries, first_move, verbose):
        if self.player_1_type == "user":
            self.user_move(available_moves, verbose, mark=1)
        elif self.player_1_type == "computer":
            self.computer_move(available_moves, mark=1)
        elif self.player_1_type == "model":
            self.model_move(self.player_1_model, available_moves, force_legal, max_retries, first_move, verbose, mark=1)
        
    def player_2_move(self, available_moves, force_legal, max_retries, first_move, verbose):
        if self.player_2_type == "user":
            self.user_move(available_moves, verbose, mark=-1)
        elif self.player_2_type == "computer":
            self.computer_move(available_moves, mark=-1)
        elif self.player_2_type == "model":
            self.model_move(self.player_2_model, available_moves, force_legal, max_retries, first_move, verbose, mark=-1)
    
    def user_move(self, available_moves, verbose, mark):
        from IPython.display import clear_output
        self.show_game_state(user_input=True)
        chosen_move_input = input(f"Enter your move (A1, B1, C3...) ")
        mapping = {"A1": 0, "A2": 1, "A3": 2,
                   "B1": 3, "B2": 4, "B3": 5,
                   "C1": 6, "C2": 7, "C3": 8}
        chosen_move = mapping.get(chosen_move_input.upper(), -1)
        clear_output(wait=True)
        while chosen_move not in available_moves:
            print(f"Invalid move. Available moves: {available_moves}")
            chosen_move_input = input(f"Enter your move (A1, B1, C3...) ")
            chosen_move = mapping.get(chosen_move_input.upper(), -1)
        self.game_state[chosen_move] = mark
        self.chosen_move = chosen_move

    def computer_move(self, available_moves, mark):
        chosen_move = np.random.choice(available_moves)
        self.game_state[chosen_move] = mark
        self.chosen_move = chosen_move
    
    def model_move(self, model, available_moves, force_legal, max_retries, first_move, verbose, mark):
        decision_made = False
        
        if not force_legal: # If there is no requirement for legal moves, just take the model's prediction without checking
            chosen_move = self.model_predict_next_move(model)
            decision_made = True

        # If it's the first move, we can allow any move to avoid biasing the model's performance based on the first move.
        if first_move:
            chosen_move = np.random.choice(available_moves)
            decision_made = True
        
        num_retries = 0
        while not decision_made:
            chosen_move = self.model_predict_next_move(model)
            decision_made = chosen_move in available_moves
            if not decision_made and verbose:
                print(f"Model predicted invalid move {chosen_move}. Retrying...")
            num_retries += 1
            if num_retries > max_retries:
                if verbose:
                    print(f"Model failed to produce a legal move after {max_retries} retries. Choosing randomly from available moves.")
                self.game_sequence["illegal_moves"].append((self.game_state.copy(), mark, chosen_move))
                chosen_move = np.random.choice(available_moves)
                decision_made = True 

        self.game_state[chosen_move] = mark
        self.chosen_move = chosen_move
    
    def model_predict_next_move(self, model):
        import torch
        model.eval()
        with torch.no_grad():
            input_tensor = torch.tensor(self.game_state, dtype=torch.float32).unsqueeze(0)
            output = model(input_tensor)
            predicted_move = torch.argmax(output, dim=1).item()
        return predicted_move

    def minimax_lookup(self):
        game_state = self.game_state
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

    def finalize_game_sequence(self, result, verbose, visualise=False):
        from IPython.display import clear_output
        from time import sleep
        if self.player_1_type == "user" or self.player_2_type == "user":
            sleep(1)
            clear_output(wait=True)
            title_text = f"Game Over!"
            self.show_game_state(user_input=False, title=title_text)
        elif verbose:
            self.show_game_state()
        if result == 3:
            if verbose or self.player_1_type == "user" or self.player_2_type == "user":
                print(f"{self.player_1_name} wins!")
        elif result == -3:
            if verbose or self.player_1_type == "user" or self.player_2_type == "user":
                print(f"{self.player_2_name} wins!")
        else:
            if verbose or self.player_1_type == "user" or self.player_2_type == "user":
                print("It's a draw!")
        who_won = 1 if result == 3 else (-1 if result == -3 else 0)
        first_player_won = True if who_won == (1 if self.game_sequence["first_player"] == self.player_1_name else -1) else False
        self.game_sequence["who_won"] = self.player_1_name if who_won == 1 else (self.player_2_name if who_won == -1 else "Draw :(")
        self.game_sequence["all_game_states"].append(self.game_state.copy())
        if visualise:
            self.show_game_sequence(persist_final_state=True)
        return self.game_sequence
    
    ## Utility methods to visualise game state and sequence
    
    def show_game_state(self, use_game_state=None, user_input=False, title=None):
        import pandas as pd
        from tabulate import tabulate
        game_state = self.game_state if use_game_state is None else use_game_state
        assert len(game_state) == 9, "Game state must have exactly 9 elements."
        
        if user_input:
            user_player_1_or_2 = (self.player_1_type == "user" or self.player_2_type == "user")
            mark = "X" if self.player_1_type == "user" else "O"
            print(f"Current game state (you are '{mark}'):")
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
        
    def show_game_sequence(self, persist_final_state=True):
        from IPython.display import clear_output
        from time import sleep
        game_sequence = self.game_sequence
        players = self.players
        for i, game_state in enumerate(game_sequence["all_game_states"][:-1]):
            if i == 0:
                title_text = f"{game_sequence['first_player']} played first:"
                sleep(1)
                self.show_game_state(use_game_state=game_state, title=title_text)
            else:
                title_text = f"{current_player} played:"
                sleep(1)
                self.show_game_state(use_game_state=game_state, title=title_text)
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
            self.show_game_state(use_game_state=game_sequence["all_game_states"][-1], title=title_text)
            
    def is_game_over(self):
        game_state = self.game_state
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

def user_vs_model(userName, opponent):
    player_1 = "user"
    if opponent == "Alice":
        player_2 = "src/models/Alice.pt"
    elif opponent == "Bob":
        player_2 = "src/models/Bob.pt"
    elif opponent == "Expert":
        player_2 = "src/models/Expert.pt"
    player_1_name = userName # by default it is "Player 1 [X]"
    player_2_name = opponent 
    game = TicTacToeGame(
        player_1, \
        player_2, \
        player_1_name=player_1_name, \
        player_2_name=player_2_name
    )
    result = game.play(visualise=False)
    return result

def evaluate_models(model_1_path, model_2_path, num_games=100, verbose=False):
    results = {"model_1_wins": 0, "model_2_wins": 0, "draws": 0}
    for i in range(num_games):
        game = TicTacToeGame(player_1=model_1_path, player_2=model_2_path)
        result = game.play(verbose=verbose)
        if result["who_won"] == game.player_1_name:
            results["model_1_wins"] += 1
        elif result["who_won"] == game.player_2_name:
            results["model_2_wins"] += 1
        else:
            results["draws"] += 1
    return results

def plot_histogram(results, model_1_name="Model 1", model_2_name="Model 2"):
    import seaborn as sns
    import matplotlib.pyplot as plt
    xlabels = [model_1_name, model_2_name, "Draws"]
    counts = [results["model_1_wins"], results["model_2_wins"], results["draws"]]
    percentages = [count / sum(counts) * 100 for count in counts]
    sns.barplot(x=xlabels, y=percentages)
    plt.ylabel("Percentage of Games Won (%)")
    plt.yticks([0, 25, 50, 75, 100])
    plt.title(f"Model Performance Comparison")
    for i, (count, percentage) in enumerate(zip(counts, percentages)):
        plt.text(i, percentage + 1, f"{count} wins\n({percentage:.1f}%)", ha='center')

def map_index_to_move(index):
    mapping = {0: "A1", 1: "A2", 2: "A3",
               3: "B1", 4: "B2", 5: "B3",
               6: "C1", 7: "C2", 8: "C3"}
    return mapping.get(index, "Invalid move index")
