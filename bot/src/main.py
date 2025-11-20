import sys
import os

# Add parent folder (bot/) to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(BOT_DIR)

WEIGHTS_PATH = os.path.join(CURRENT_DIR, "data_generation", "chess_cnn_eval.pt")

import chess
from chess import Move
import random
import time
import torch

from src.data_generation.train_cnn import fen_to_tensor, ChessEvalDataset, ChessCNN
from src.utils import chess_manager, GameContext
 
# Write code here that runs once
# Can do things like load models from huggingface, make connections to subprocesses, etcwenis
 
inf = 100_000
TIME_LIMIT_SECONDS = 3.0  # target: ~3 seconds per move
CONSIDERED_MOVES_DEBUG: list[str] = []

# =========================
#  Material values + helpers
# =========================

PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  950,
    chess.KING:   0,
}

# max material loss we will allow a move to cause (in centipawns)
# ~400 = about a minor piece
MAX_ALLOWED_LOSS = 150


def material_score(board: chess.Board, color: bool) -> int:
    """Total material for given color, in centipawns."""
    score = 0
    for piece_type, value in PIECE_VALUES.items():
        score += len(board.pieces(piece_type, color)) * value
    return score


def is_move_too_bad(board: chess.Board, move: chess.Move, color: bool) -> bool:
    """
    Returns True if the move loses more than MAX_ALLOWED_LOSS material
    for `color` in a static one-ply sense.
    """
    material_before = material_score(board, color)

    board.push(move)
    material_after = material_score(board, color)
    board.pop()

    loss = material_before - material_after
    return loss > MAX_ALLOWED_LOSS


class NNChessEvaluator:
    def __init__(self, weights_path="chess_cnn_eval.pt", device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
 
        self.model = ChessCNN(in_channels=18).to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()
 
    @torch.no_grad()
    def eval_fen(self, fen: str) -> float:
        """
        Returns evaluation in centipawns, from the side to move's perspective.
        """
        x_np = fen_to_tensor(fen)                    # (18,8,8) np
        x = torch.from_numpy(x_np).unsqueeze(0)      # (1,18,8,8)
        x = x.to(self.device)
 
        out = self.model(x)                          # (1,1) normalized
        eval_normalized = out.item()                 # float
 
        # Undo the normalization used in training
        eval_cp = eval_normalized * 400.0            # back to centipawns
        return eval_cp
 
    @torch.no_grad()
    def eval_board(self, board: chess.Board) -> float:
        """
        Convenience: take a python-chess Board instead of FEN.
        """
        fen = board.fen()
        return self.eval_fen(fen)
 
evaluator = NNChessEvaluator(WEIGHTS_PATH, device="cpu")
 
# =========================
# NEW FIXED EVAL + SEARCH USING NN (WITH MAXIMIZING_COLOR)
# =========================
 
def evaluate(board: chess.Board, maximizing_color: bool) -> float:
    """
    Returns evaluation from the perspective of `maximizing_color`.
    Uses NN (side-to-move eval) and converts sign when needed.
    """
 
    # --- Terminal conditions ---
    if board.is_checkmate():
        # Side to move is checkmated
        if board.turn == maximizing_color:
            return -inf
        else:
            return inf
 
    # "Hard" draws only
    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.is_seventyfive_moves()
    ):
        return 0.0
 
    # --- Non-terminal: use NN eval from side-to-move perspective ---
    stm_eval = evaluator.eval_board(board)  # >0 good for side to move
 
    # Convert to maximizing_color perspective
    if board.turn == maximizing_color:
        return stm_eval
    else:
        return -stm_eval
 
 
def alphabeta(
    board: chess.Board,
    depth: int,
    alpha: float,
    beta: float,
    maximizing_player: bool,
    maximizing_color: bool,
    search_start: float,
):
    """
    Alpha-beta pruning on a python-chess Board, using NN evaluation.
    Evaluation is always interpreted from the perspective of `maximizing_color`.
    """
    # Time cutoff — return static eval from this node
    if time.time() - search_start >= TIME_LIMIT_SECONDS:
        return None, evaluate(board, maximizing_color)

    if depth == 0 or board.is_game_over():
        return None, evaluate(board, maximizing_color)
 
    best_move: Move | None = None
 
    if maximizing_player:
        value = -inf

        # collect legal moves once so we can handle "all moves filtered" case
        legal_moves = list(board.generate_legal_moves())
        filtered_moves: list[Move] = []
        for m in legal_moves:
            if not is_move_too_bad(board, m, maximizing_color):
                filtered_moves.append(m)
                CONSIDERED_MOVES_DEBUG.append(str(m))
        if not filtered_moves:
            # if everything is "too bad", we still have to search something
            filtered_moves = legal_moves

        for move in filtered_moves:
            board.push(move)
            _, score = alphabeta(
                board,
                depth - 1,
                alpha,
                beta,
                False,               # opponent is minimizing
                maximizing_color,
                search_start,
            )
            board.pop()
 
            if best_move is None or score > value:
                value = score
                best_move = move
 
            alpha = max(alpha, value)
            if alpha >= beta:
                break  # beta cut-off
 
        return best_move, value
 
    else:
        value = inf

        legal_moves = list(board.generate_legal_moves())
        filtered_moves: list[Move] = []
        for m in legal_moves:
            if not is_move_too_bad(board, m, maximizing_color):
                filtered_moves.append(m)
                CONSIDERED_MOVES_DEBUG.append(str(m))
        if not filtered_moves:
            filtered_moves = legal_moves

        for move in filtered_moves:
            board.push(move)
            _, score = alphabeta(
                board,
                depth - 1,
                alpha,
                beta,
                True,                # opponent is maximizing
                maximizing_color,
                search_start,
            )
            board.pop()
 
            if best_move is None or score < value:
                value = score
                best_move = move
 
            beta = min(beta, value)
            if beta <= alpha:
                break  # alpha cut-off
 
        return best_move, value
 
 
@chess_manager.entrypoint
def test_func(ctx: GameContext) -> Move:
    # This gets called every time the model needs to make a move
    # Must return a python-chess Move object that is a legal move
 
    print("Cooking move...")
    print(ctx.board.move_stack)
    time.sleep(0.1)
 
    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available (I probably lost, didn't I)")
 
    # Root side to move = maximizing color
    maximizing_color = ctx.board.turn

    # Start timer for this move
    search_start = time.time()
 
    # Search with alpha-beta
    search_depth = 4  # max depth; time limit will cut this off if needed
    best_move, best_eval = alphabeta(
        ctx.board,
        depth=search_depth,
        alpha=-inf,
        beta=inf,
        maximizing_player=True,        # side to move is maximizing
        maximizing_color=maximizing_color,
        search_start=search_start,
    )

    # Debug: send considered moves to frontend
    if CONSIDERED_MOVES_DEBUG:
        ctx.logProbabilities({
            f"considered_{i}: {mv}": 0.0
            for i, mv in enumerate(CONSIDERED_MOVES_DEBUG)
        })
        CONSIDERED_MOVES_DEBUG.clear()
 
    if best_move is None:
        # Fallback: should basically never happen unless depth == 0
        best_move = random.choice(legal_moves)
 
    # Build probabilities: dict[Move, float] — exactly what serve.py expects
    move_probs: dict[Move, float] = {m: 0.0 for m in legal_moves}
    move_probs[best_move] = 1.0  # one-hot on best move (float)
 
    ctx.logProbabilities(move_probs)
 
    # IMPORTANT: test_func returns ONLY a Move.
    return best_move
 
@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    pass