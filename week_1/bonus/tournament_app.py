"""
Tic-Tac-Toe Model Tournament App
Requires: pip install PyQt5
"""

import sys
import os
import random
import numpy as np
from pathlib import Path
from itertools import combinations

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QTextEdit,
    QDialog, QGridLayout, QFrame, QGroupBox, QMessageBox, QSizePolicy,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor, QPalette

# Make sure src/ imports work when running from week_1/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game_utils import TicTacToeGame, map_index_to_move


# ---------------------------------------------------------------------------
# Tournament logic helpers  (unchanged)
# ---------------------------------------------------------------------------

def _evaluate_models(model_1_path, model_2_path, num_games=100):
    results = {"model_1_wins": 0, "model_2_wins": 0, "draws": 0}
    for _ in range(num_games):
        game = TicTacToeGame(player_1=model_1_path, player_2=model_2_path)
        result = game.play(verbose=False)
        if result["who_won"] == game.player_1_name:
            results["model_1_wins"] += 1
        elif result["who_won"] == game.player_2_name:
            results["model_2_wins"] += 1
        else:
            results["draws"] += 1
    return results


def _play_group_of_2(m1, m2, num_games=100):
    results = _evaluate_models(m1, m2, num_games)
    if results["model_1_wins"] > results["model_2_wins"]:
        return m1, results
    elif results["model_2_wins"] > results["model_1_wins"]:
        return m2, results
    else:
        return random.choice([m1, m2]), results


def _play_group_of_3(paths, num_games=100):
    scores = {p: 0 for p in paths}
    match_log = []
    for m1, m2 in combinations(paths, 2):
        winner, results = _play_group_of_2(m1, m2, num_games)
        scores[winner] += 1
        match_log.append((m1, m2, winner, results))
    best = max(scores.values())
    winners = [p for p, s in scores.items() if s == best]
    return random.choice(winners), scores, match_log


def _make_groups(models):
    n = len(models)
    groups = []
    if n % 2 == 0:
        for i in range(0, n, 2):
            groups.append(models[i : i + 2])
    else:
        for i in range(0, n - 3, 2):
            groups.append(models[i : i + 2])
        groups.append(models[n - 3 :])
    return groups


# ---------------------------------------------------------------------------
# Background worker  (unchanged)
# ---------------------------------------------------------------------------

class TournamentWorker(QThread):
    log_msg     = pyqtSignal(str)
    round_done  = pyqtSignal(int, list)
    final_ready = pyqtSignal(str, str)
    error       = pyqtSignal(str)

    NUM_GAMES = 100

    def __init__(self, folder: str):
        super().__init__()
        self.folder = folder

    def _log(self, msg: str):
        self.log_msg.emit(msg)

    def run(self):
        try:
            folder = Path(self.folder)
            paths = sorted(folder.glob("*.pt"))

            if len(paths) < 2:
                self.error.emit("Need at least 2 .pt model files in the folder.")
                return
            if len(paths) > 50:
                self.error.emit("Too many models — maximum is 50.")
                return

            self._log(f"Found {len(paths)} models:")
            for p in paths:
                self._log(f"  • {p.stem}")

            current = [str(p) for p in paths]
            random.shuffle(current)
            round_num = 0

            while len(current) > 3:
                round_num += 1
                self._log(f"\n{'═'*44}")
                self._log(f"  ROUND {round_num}  —  {len(current)} models")
                self._log(f"{'═'*44}")
                random.shuffle(current)
                winners = []

                for group in _make_groups(current):
                    if len(group) == 2:
                        m1, m2 = group
                        n1, n2 = Path(m1).stem, Path(m2).stem
                        self._log(f"\n  ▸ {n1}  vs  {n2}")
                        winner, res = _play_group_of_2(m1, m2, self.NUM_GAMES)
                        wname = Path(winner).stem
                        self._log(
                            f"    Result: {res['model_1_wins']}-{res['model_2_wins']}-{res['draws']}"
                            f"  →  Winner: {wname}"
                        )
                        winners.append(winner)
                    else:
                        names = [Path(p).stem for p in group]
                        self._log(f"\n  ▸ Round-robin: {', '.join(names)}")
                        winner, scores, _ = _play_group_of_3(group, self.NUM_GAMES)
                        wname = Path(winner).stem
                        score_str = "  |  ".join(
                            f"{Path(p).stem}: {s}" for p, s in scores.items()
                        )
                        self._log(f"    Scores: {score_str}  →  Winner: {wname}")
                        winners.append(winner)

                winner_names = [Path(w).stem for w in winners]
                self.round_done.emit(round_num, winner_names)
                self._log(f"\n  Advancing: {', '.join(winner_names)}")
                current = winners

            if len(current) == 3:
                names = [Path(p).stem for p in current]
                self._log(f"\n3 models remain ({', '.join(names)}).")
                self._log("Randomly selecting 2 for the final…")
                idx = list(np.random.choice(3, 2, replace=False))
                current = [current[i] for i in idx]

            f1, f2 = current[0], current[1]
            self._log(f"\n{'═'*44}")
            self._log(f"  FINAL:  {Path(f1).stem}  vs  {Path(f2).stem}")
            self._log(f"{'═'*44}")
            self.final_ready.emit(f1, f2)

        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Probability helpers
# ---------------------------------------------------------------------------

def _prob_color(p: float) -> str:
    """Map probability [0, 1] → colour from dim grey to bright gold."""
    t = min(p * 2.5, 1.0)          # amplify so ~40 % → full gold
    r = int(0x33 + (0xff - 0x33) * t)
    g = int(0x33 + (0xd7 - 0x33) * t)
    b = int(0x33 + (0x00 - 0x33) * t)
    return f"#{max(0,min(255,r)):02x}{max(0,min(255,g)):02x}{max(0,min(255,b)):02x}"


def _run_model(game_state: np.ndarray, model) -> np.ndarray:
    """Return 9-element softmax probability vector from a model."""
    import torch
    model.eval()
    with torch.no_grad():
        t = torch.tensor(game_state.astype(np.float32)).unsqueeze(0)
        probs = model(t).squeeze().detach().numpy()
    return probs


# ---------------------------------------------------------------------------
# Cell widget  — symbol (large) + probability % (small), dark theme
# ---------------------------------------------------------------------------

class CellWidget(QFrame):
    _BG_EMPTY   = "#111827"
    _BG_X       = "#1a0505"
    _BG_X_HI    = "#2d0a0a"
    _BG_O       = "#050518"
    _BG_O_HI    = "#0a0a2d"
    _COL_X      = "#ef4444"
    _COL_X_HI   = "#ff8080"
    _COL_O      = "#60a5fa"
    _COL_O_HI   = "#93c5fd"

    def __init__(self):
        super().__init__()
        self.setObjectName("cell")
        self.setFixedSize(122, 122)
        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(4, 6, 4, 4)
        vbox.setSpacing(1)

        self._sym = QLabel("")
        self._sym.setObjectName("symLabel")
        self._sym.setAlignment(Qt.AlignCenter)
        self._sym.setFont(QFont("Arial", 40, QFont.Bold))

        self._prob = QLabel("")
        self._prob.setObjectName("probLabel")
        self._prob.setAlignment(Qt.AlignCenter)
        self._prob.setFont(QFont("Arial", 11, QFont.Bold))

        vbox.addWidget(self._sym, stretch=4)
        vbox.addWidget(self._prob, stretch=1)

        self.set_state(0, None, False)

    def set_state(self, value: int, prob: float | None, highlighted: bool):
        sym_text = {0: "", 1: "X", -1: "O"}[value]
        self._sym.setText(sym_text)

        if value == 1:
            sym_color = self._COL_X_HI if highlighted else self._COL_X
            bg        = self._BG_X_HI  if highlighted else self._BG_X
            border    = self._COL_X_HI if highlighted else "#5a1010"
        elif value == -1:
            sym_color = self._COL_O_HI if highlighted else self._COL_O
            bg        = self._BG_O_HI  if highlighted else self._BG_O
            border    = self._COL_O_HI if highlighted else "#10105a"
        else:
            sym_color = "#555"
            bg        = self._BG_EMPTY
            # Tint border gold when the model gives this cell high probability
            if prob is not None and prob > 0.20:
                t = min((prob - 0.20) / 0.30, 1.0)
                ri = int(0x2d + (0xb8 - 0x2d) * t)
                gi = int(0x37 + (0x86 - 0x37) * t)
                bi = int(0x48 + (0x00 - 0x48) * t)
                border = f"#{ri:02x}{gi:02x}{bi:02x}"
            else:
                border = "#2d3748"

        self.setStyleSheet(
            f"QFrame#cell {{ background-color: {bg};"
            f" border: 2px solid {border}; border-radius: 8px; }}"
        )
        self._sym.setStyleSheet(
            f"QLabel#symLabel {{ color: {sym_color}; background: transparent; }}"
        )

        if prob is not None:
            color = _prob_color(prob)
            self._prob.setText(f"{prob * 100:.0f}%")
            self._prob.setStyleSheet(
                f"QLabel#probLabel {{ color: {color}; background: transparent; }}"
            )
        else:
            self._prob.setText("")
            self._prob.setStyleSheet(
                "QLabel#probLabel { background: transparent; }"
            )


# ---------------------------------------------------------------------------
# Board widget
# ---------------------------------------------------------------------------

class BoardWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: #0d0f18;")
        layout = QGridLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)

        self._cells: list[list[CellWidget]] = []
        for r in range(3):
            row = []
            for c in range(3):
                cell = CellWidget()
                layout.addWidget(cell, r, c)
                row.append(cell)
            self._cells.append(row)

    def update_board(self, game_state, probs=None, highlight_idx: int = -1):
        for idx in range(9):
            r, c = idx // 3, idx % 3
            val  = int(game_state[idx])
            p    = float(probs[idx]) if probs is not None else None
            self._cells[r][c].set_state(val, p, highlighted=(idx == highlight_idx))

    def clear(self):
        self.update_board(np.zeros(9, dtype=int))


# ---------------------------------------------------------------------------
# Final match window
# ---------------------------------------------------------------------------

class FinalMatchWindow(QDialog):
    champion_decided = pyqtSignal(str)

    _STEP_MS     = 1400   # ms between auto-play steps
    _MAX_RETRIES = 20     # max games to play before accepting a draw

    _DARK_BG  = "#0d0f18"
    _DARK_MID = "#161824"

    def __init__(self, m1_path: str, m2_path: str, parent=None):
        super().__init__(parent)
        self.m1_path = m1_path
        self.m2_path = m2_path
        self.m1_name = Path(m1_path).stem
        self.m2_name = Path(m2_path).stem

        self._display: list[tuple] = []   # (state, title, probs | None, hi_idx)
        self.current_step = 0
        self._finished = False

        self._timer = QTimer()
        self._timer.timeout.connect(self._advance)

        self.setWindowTitle(f"FINAL  ·  {self.m1_name}  vs  {self.m2_name}")
        self.setModal(True)
        self.setStyleSheet(f"QDialog {{ background-color: {self._DARK_BG}; }}")

        self._build_ui()
        self._play_game()

    # ------------------------------------------------------------------
    def _build_ui(self):
        label_style = "color: #e2e8f0; background: transparent;"
        root = QVBoxLayout(self)
        root.setSpacing(14)
        root.setContentsMargins(22, 22, 22, 22)

        # Title
        self._title = QLabel("Preparing final match…")
        self._title.setAlignment(Qt.AlignCenter)
        self._title.setFont(QFont("Arial", 19, QFont.Bold))
        self._title.setStyleSheet("color: #f8fafc; background: transparent;")
        root.addWidget(self._title)

        # Player legend
        legend = QHBoxLayout()
        x_lbl = QLabel(f"✕  {self.m1_name}")
        x_lbl.setFont(QFont("Arial", 13, QFont.Bold))
        x_lbl.setStyleSheet("color: #ef4444; background: transparent;")
        o_lbl = QLabel(f"◯  {self.m2_name}")
        o_lbl.setFont(QFont("Arial", 13, QFont.Bold))
        o_lbl.setStyleSheet("color: #60a5fa; background: transparent;")
        legend.addWidget(x_lbl)
        legend.addStretch()
        legend.addWidget(o_lbl)
        root.addLayout(legend)

        # Board
        self._board = BoardWidget()
        root.addWidget(self._board, alignment=Qt.AlignCenter)

        # Probability legend
        prob_legend = QLabel(
            "Cell numbers = model's predicted probability for each square  "
            "  (dim → gold = low → high)"
        )
        prob_legend.setAlignment(Qt.AlignCenter)
        prob_legend.setFont(QFont("Arial", 9))
        prob_legend.setStyleSheet("color: #64748b; background: transparent;")
        prob_legend.setWordWrap(True)
        root.addWidget(prob_legend)

        # Status line
        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignCenter)
        self._status.setFont(QFont("Arial", 13))
        self._status.setStyleSheet(label_style)
        self._status.setWordWrap(True)
        root.addWidget(self._status)

        # Buttons
        btn_row = QHBoxLayout()
        btn_style = (
            "QPushButton { background-color: #1e2433; color: #e2e8f0;"
            " border: 1px solid #2d3748; border-radius: 6px;"
            " padding: 7px 18px; font-size: 12px; }"
            "QPushButton:hover { background-color: #2d3748; }"
            "QPushButton:disabled { color: #4a5568; border-color: #1e2433; }"
        )
        self._next_btn = QPushButton("▶  Next Move")
        self._next_btn.setEnabled(False)
        self._next_btn.setStyleSheet(btn_style)
        self._next_btn.clicked.connect(self._advance)

        self._auto_btn = QPushButton("⏩  Auto Play")
        self._auto_btn.setEnabled(False)
        self._auto_btn.setStyleSheet(btn_style)
        self._auto_btn.clicked.connect(self._toggle_auto)

        btn_row.addWidget(self._next_btn)
        btn_row.addWidget(self._auto_btn)
        root.addLayout(btn_row)

        self._close_btn = QPushButton("Close")
        self._close_btn.setStyleSheet(btn_style)
        self._close_btn.setVisible(False)
        self._close_btn.clicked.connect(self.accept)
        root.addWidget(self._close_btn)

        self.setMinimumWidth(560)
        self.resize(560, 660)

    # ------------------------------------------------------------------
    def _play_until_winner(self):
        """Play games until a non-draw result (max _MAX_RETRIES attempts)."""
        draws = 0
        game = seq = None
        for attempt in range(1, self._MAX_RETRIES + 1):
            game = TicTacToeGame(player_1=self.m1_path, player_2=self.m2_path)
            seq  = game.play(verbose=False)
            if seq["who_won"] != "Draw :(":
                return game, seq, draws
            draws += 1
            self._status.setText(f"Draw #{draws} — replaying to find a winner…")
            QApplication.processEvents()
        # Exhausted retries — return the last (draw) game
        return game, seq, draws

    # ------------------------------------------------------------------
    def _prepare_display(self, seq, model_1, model_2) -> list[tuple]:
        """
        Build the sequence of (state, title, probs, highlight_idx) to show.

        all_game_states layout (from TicTacToeGame):
          [state_after_move_1, ..., state_after_last_move, duplicate_final]
        all_user_inputs layout:
          [(empty, 0, -1),  (board, player, move), ...]
          index 0 = sentinel; index i+1 = after move i+1

        Probabilities shown are for the player who is *about to move next*
        on the current board — i.e. what they are "thinking".
        At game-over (last board) no probabilities are shown.
        """
        all_states = seq["all_game_states"]   # includes duplicate tail
        all_inputs = seq["all_user_inputs"]
        players    = {1: seq["players"][1], -1: seq["players"][-1]}

        first_player_name = seq["first_player"]
        first_mark = 1 if "[X]" in first_player_name else -1
        mark_to_model = {1: model_1, -1: model_2}

        n_moves = len(all_states) - 1          # real move count

        display = []

        # Step 0: empty board — show first player's probability vector
        empty = np.zeros(9, dtype=int)
        probs_0 = _run_model(empty, mark_to_model[first_mark])
        display.append((
            empty,
            f"Board is empty  —  {first_player_name} plays first",
            probs_0,
            -1,
        ))

        # Steps 1 … n_moves
        for i in range(n_moves):
            state = all_states[i]          # board after the (i+1)-th move
            inp   = all_inputs[i + 1]      # (board_before, player_mark, move_idx)
            player_val  = inp[1]
            move_idx    = inp[2]
            player_name = players.get(player_val, "?")
            coord       = map_index_to_move(move_idx)
            title       = f"{player_name}  played  {coord}"

            is_last = (i == n_moves - 1)
            if is_last:
                probs = None               # game over — no next mover
            else:
                next_mark = -player_val
                probs = _run_model(state, mark_to_model[next_mark])

            display.append((state, title, probs, move_idx))

        return display

    # ------------------------------------------------------------------
    def _play_game(self):
        self._title.setText("Playing final game…")
        QApplication.processEvents()

        game, seq, draws_skipped = self._play_until_winner()

        # Load model objects for probability inference
        from src.train_game_utils import load_model_from_path
        model_1 = load_model_from_path(self.m1_path)
        model_2 = load_model_from_path(self.m2_path)

        self._seq     = seq
        self._players = game.players
        self._winner  = seq["who_won"]
        self._champion_path = (
            self.m1_path if seq["who_won"] == game.player_1_name
            else self.m2_path if seq["who_won"] == game.player_2_name
            else None
        )

        self._display = self._prepare_display(seq, model_1, model_2)

        skip_note = f"  ({draws_skipped} draw(s) skipped)" if draws_skipped else ""
        self._title.setText(
            f"FINAL  ·  {self.m1_name}  vs  {self.m2_name}{skip_note}"
        )
        self._status.setText(
            "Press  ▶ Next Move  or  ⏩ Auto Play  to watch the game."
        )
        self._board.clear()
        self._next_btn.setEnabled(True)
        self._auto_btn.setEnabled(True)

    # ------------------------------------------------------------------
    def _advance(self):
        if self.current_step >= len(self._display):
            self._finish()
            return

        state, title, probs, hi_idx = self._display[self.current_step]
        self._board.update_board(state, probs, hi_idx)
        self._status.setText(title)
        self.current_step += 1

        if self.current_step >= len(self._display):
            self._finish()

    def _finish(self):
        if self._finished:
            return
        self._finished = True
        self._timer.stop()
        self._next_btn.setEnabled(False)
        self._auto_btn.setEnabled(False)

        if self._winner == "Draw :(":
            self._title.setText("Draw  —  picking champion at random")
            champ = random.choice([self.m1_name, self.m2_name])
        else:
            self._title.setStyleSheet("color: #fbbf24; background: transparent;")
            self._title.setText(f"🏆  {self._winner}  WINS!")
            champ = (
                Path(self._champion_path).stem
                if self._champion_path
                else random.choice([self.m1_name, self.m2_name])
            )

        self._status.setText(f"Tournament Champion:  {champ}")
        self._close_btn.setVisible(True)
        self.champion_decided.emit(champ)

    def _toggle_auto(self):
        if self._timer.isActive():
            self._timer.stop()
            self._auto_btn.setText("⏩  Auto Play")
            self._next_btn.setEnabled(True)
        else:
            self._timer.start(self._STEP_MS)
            self._auto_btn.setText("⏸  Pause")
            self._next_btn.setEnabled(False)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tic-Tac-Toe  ·  Model Tournament")
        self.resize(860, 700)
        self._worker: TournamentWorker | None = None
        self._build_ui()

    def _build_ui(self):
        root_widget = QWidget()
        self.setCentralWidget(root_widget)
        root = QVBoxLayout(root_widget)
        root.setSpacing(12)
        root.setContentsMargins(16, 16, 16, 16)

        title_lbl = QLabel("Tic-Tac-Toe  Model Tournament")
        title_lbl.setAlignment(Qt.AlignCenter)
        title_lbl.setFont(QFont("Arial", 22, QFont.Bold))
        root.addWidget(title_lbl)

        folder_box = QGroupBox("Models Folder")
        folder_layout = QHBoxLayout(folder_box)

        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText(
            "Path to folder containing  .pt  model files…"
        )
        self._folder_edit.setFont(QFont("Arial", 11))
        folder_layout.addWidget(self._folder_edit)

        browse_btn = QPushButton("Browse…")
        browse_btn.setFont(QFont("Arial", 11))
        browse_btn.clicked.connect(self._browse)
        folder_layout.addWidget(browse_btn)

        self._start_btn = QPushButton("▶  Start Tournament")
        self._start_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self._start_btn.setStyleSheet(
            "QPushButton { background-color: #16a34a; color: white;"
            " padding: 6px 14px; border-radius: 5px; }"
            "QPushButton:disabled { background-color: #374151; color: #6b7280; }"
        )
        self._start_btn.clicked.connect(self._start)
        folder_layout.addWidget(self._start_btn)

        root.addWidget(folder_box)

        log_box = QGroupBox("Tournament Progress")
        log_layout = QVBoxLayout(log_box)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Courier New", 11))
        self._log.setStyleSheet(
            "QTextEdit { background-color: #111827; color: #d1d5db; }"
        )
        log_layout.addWidget(self._log)
        root.addWidget(log_box)

        self._champion_lbl = QLabel("")
        self._champion_lbl.setAlignment(Qt.AlignCenter)
        self._champion_lbl.setFont(QFont("Arial", 18, QFont.Bold))
        self._champion_lbl.setStyleSheet("color: #fbbf24;")
        root.addWidget(self._champion_lbl)

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Models Folder")
        if folder:
            self._folder_edit.setText(folder)

    def _start(self):
        folder = self._folder_edit.text().strip()
        if not folder:
            QMessageBox.warning(self, "No Folder", "Please select a models folder first.")
            return
        if not os.path.isdir(folder):
            QMessageBox.warning(self, "Invalid Folder", "That folder does not exist.")
            return

        self._log.clear()
        self._champion_lbl.setText("")
        self._start_btn.setEnabled(False)

        self._worker = TournamentWorker(folder)
        self._worker.log_msg.connect(self._append)
        self._worker.round_done.connect(self._on_round_done)
        self._worker.final_ready.connect(self._on_final_ready)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(lambda: self._start_btn.setEnabled(True))
        self._worker.start()

    def _append(self, msg: str):
        self._log.append(msg)
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_round_done(self, round_num: int, winner_names: list):
        self._append(f"\n  ✓ Round {round_num} complete — advancing: {', '.join(winner_names)}")

    def _on_final_ready(self, m1_path: str, m2_path: str):
        m1 = Path(m1_path).stem
        m2 = Path(m2_path).stem
        self._append(f"\n  Launching FINAL:  {m1}  vs  {m2}")
        dlg = FinalMatchWindow(m1_path, m2_path, self)
        dlg.champion_decided.connect(self._on_champion)
        dlg.exec_()

    def _on_champion(self, name: str):
        self._champion_lbl.setText(f"🏆  TOURNAMENT CHAMPION:  {name}  🏆")
        self._append(f"\n{'═'*44}")
        self._append(f"  CHAMPION:  {name}")
        self._append(f"{'═'*44}")

    def _on_error(self, msg: str):
        QMessageBox.critical(self, "Tournament Error", msg)
        self._start_btn.setEnabled(True)


# ---------------------------------------------------------------------------
# Entry point — dark Fusion palette applied app-wide
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    c = QColor
    p = QPalette()
    p.setColor(QPalette.Window,          c(15,  17,  24))
    p.setColor(QPalette.WindowText,      c(226, 232, 240))
    p.setColor(QPalette.Base,            c(17,  24,  39))
    p.setColor(QPalette.AlternateBase,   c(30,  36,  51))
    p.setColor(QPalette.ToolTipBase,     c(15,  17,  24))
    p.setColor(QPalette.ToolTipText,     c(226, 232, 240))
    p.setColor(QPalette.Text,            c(226, 232, 240))
    p.setColor(QPalette.Button,          c(30,  36,  51))
    p.setColor(QPalette.ButtonText,      c(226, 232, 240))
    p.setColor(QPalette.BrightText,      c(255, 100, 100))
    p.setColor(QPalette.Link,            c(96,  165, 250))
    p.setColor(QPalette.Highlight,       c(37,  99,  235))
    p.setColor(QPalette.HighlightedText, c(255, 255, 255))
    app.setPalette(p)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
