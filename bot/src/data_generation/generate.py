import os
import random
import subprocess
from typing import List, Dict, Optional

import numpy as np
import chess
import chess.pgn

#cd my-chesshacks-bot/src/data_generation
pgn_path = "/Users/srirenukasayanthan/Desktop/chesshacks/bot/src/data_generation/lichess_elite_2025-02.pgn"

# 1. .pgn file => number of games
def count_games_in_pgn(pgn_path: str) -> int:
    """
    Counts how many games are in a PGN file.
    """
    count = 0
    with open(pgn_path, "r", encoding="utf-8") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            count += 1
            if (count % 1000 == 0):
                print(f"{count} games counted.")
    return count


# 2. .pgn game => 6 random FEN (2 openings, 2 middle, 2 endgames where possible)

def sample_fens_from_game(
    game: chess.pgn.Game,
    num_opening: int = 1,
    num_middle: int = 2,
    num_endgame: int = 1,
) -> List[str]:
    """
    Given a single chess.pgn.Game, returns up to 6 FEN strings, trying to get:
    - num_opening positions from early moves
    - num_middle from middle game
    - num_endgame from later moves

    Heuristic:
      - opening: fullmove_number <= 10
      - middlegame: 11 <= fullmove_number <= 30
      - endgame: fullmove_number >= 31
    """
    
    board = game.board()
    opening_positions = []
    middle_positions = []
    endgame_positions = []

    for move in game.mainline_moves():
        board.push(move)
        fullmove = board.fullmove_number

        if fullmove <= 10:
            opening_positions.append(board.fen())
        elif 11 <= fullmove <= 30:
            middle_positions.append(board.fen())
        else:
            endgame_positions.append(board.fen())

    def sample_from_bucket(bucket: List[str], k: int) -> List[str]:
        if len(bucket) <= k:
            return bucket.copy()
        return random.sample(bucket, k)

    # Try to sample the requested numbers
    sampled_opening = sample_from_bucket(opening_positions, num_opening)
    sampled_middle = sample_from_bucket(middle_positions, num_middle)
    sampled_end = sample_from_bucket(endgame_positions, num_endgame)

    result = sampled_opening + sampled_middle + sampled_end

    # If we didn't get 6 total because buckets were small, fill from whatever is left
    all_positions = opening_positions + middle_positions + endgame_positions
    if len(result) < (num_opening + num_middle + num_endgame):
        needed = (num_opening + num_middle + num_endgame) - len(result)
        # exclude already chosen positions
        remaining = [fen for fen in all_positions if fen not in result]
        if len(remaining) > needed:
            result.extend(random.sample(remaining, needed))
        else:
            result.extend(remaining)

    # Ensure uniqueness and cap at total 6
    # (in case of small games, you may get fewer than 6)
    result = list(dict.fromkeys(result))  # remove duplicates, preserve order
    return result[: (num_opening + num_middle + num_endgame)]


def sample_fens_from_game_to_file(
    game: chess.pgn.Game,
    file_path: str = '/Users/srirenukasayanthan/Desktop/chesshacks/bot/src/data_generation/fen_data.txt',
    num_opening: int = 1,
    num_middle: int = 2,
    num_endgame: int = 1,
):
    """
    Writes up to 6 sampled FEN positions from `game` to a file.

    File format:
        Each line = one FEN string

    Position sampling heuristic:
      - opening = fullmove_number <= 10
      - middle  = 11 <= fullmove_number <= 30
      - endgame = fullmove_number >= 31
    """
    
    board = game.board()
    opening_positions = []
    middle_positions = []
    endgame_positions = []

    # Collect all positions
    for move in game.mainline_moves():
        board.push(move)
        fullmove = board.fullmove_number

        if fullmove <= 10:
            opening_positions.append(board.fen())
        elif 11 <= fullmove <= 30:
            middle_positions.append(board.fen())
        else:
            endgame_positions.append(board.fen())

    def take(bucket, k):
        if len(bucket) <= k:
            return bucket
        return random.sample(bucket, k)

    sampled = []
    sampled.extend(take(opening_positions, num_opening))
    sampled.extend(take(middle_positions, num_middle))
    sampled.extend(take(endgame_positions, num_endgame))

    # If too few positions, fill from leftovers
    all_positions = opening_positions + middle_positions + endgame_positions
    if len(sampled) < (num_opening + num_middle + num_endgame):
        needed = (num_opening + num_middle + num_endgame) - len(sampled)
        remaining = [fen for fen in all_positions if fen not in sampled]
        if len(remaining) > needed:
            sampled.extend(random.sample(remaining, needed))
        else:
            sampled.extend(remaining)

    # Remove duplicates and trim to max desired
    sampled = list(dict.fromkeys(sampled))[: (num_opening + num_middle + num_endgame)]

    # Write to file
    with open(file_path, "a", encoding="utf-8") as f:
        for fen in sampled:
            f.write(fen + "\n")

# 3. FEN => Tensor (accounting for en passant, castling rights, turns)
#
# Promotions are naturally encoded because the promoted piece is present on the board
# (e.g. a white queen on the 8th rank that used to be a pawn). There is no special flag.


PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def fen_to_tensor(fen: str) -> np.ndarray:
    """
    Converts a FEN string into a tensor of shape (C, 8, 8) suitable for a CNN.

    Channels (C = 18):
      0-5   : white {P, N, B, R, Q, K}
      6-11  : black {P, N, B, R, Q, K}
      12    : side to move (all ones if white to move, zeros otherwise)
      13    : white can castle kingside
      14    : white can castle queenside
      15    : black can castle kingside
      16    : black can castle queenside
      17    : en-passant target square (one-hot; all zeros if none)

    Board orientation:
      [0, :, :] is the 8th rank (from White's perspective),
      [7, :, :] is the 1st rank.
    """
    board = chess.Board(fen)
    tensor = np.zeros((18, 8, 8), dtype=np.float32)

    # 12 piece planes
    for square, piece in board.piece_map().items():
        piece_type = piece.piece_type
        color = piece.color  # True=white, False=black
        if piece_type not in PIECE_TYPES:
            continue

        type_index = PIECE_TYPES.index(piece_type)
        if color == chess.WHITE:
            plane = type_index  # 0-5
        else:
            plane = 6 + type_index  # 6-11

        file = chess.square_file(square)   # 0..7
        rank = chess.square_rank(square)   # 0..7, 0 = 1st rank
        row = 7 - rank                     # row 0 = 8th rank
        col = file
        tensor[plane, row, col] = 1.0

    # Side to move
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0
    else:
        tensor[12, :, :] = 0.0  # already zero, but explicit for clarity

    # Castling rights
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[16, :, :] = 1.0

    # En passant square (if any)
    if board.ep_square is not None:
        file = chess.square_file(board.ep_square)
        rank = chess.square_rank(board.ep_square)
        row = 7 - rank
        col = file
        tensor[17, row, col] = 1.0

    return tensor


# 4. Local stockfish: position (FEN) => dict{FEN, Tensor, eval}


class StockfishUCI:
    """
    Minimal UCI wrapper around a local Stockfish binary.
    """

    def __init__(self, engine_path: str = "stockfish"):
        if not os.path.exists(engine_path) and engine_path != "stockfish":
            raise FileNotFoundError(f"Stockfish binary not found at: {engine_path}")

        self.proc = subprocess.Popen(
            [engine_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._init_uci()

    def _send(self, cmd: str):
        assert self.proc.stdin is not None
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _read_line(self) -> str:
        assert self.proc.stdout is not None
        return self.proc.stdout.readline()

    def _init_uci(self):
        self._send("uci")
        while True:
            line = self._read_line().strip()
            if line == "uciok":
                break
        self._send("isready")
        while True:
            line = self._read_line().strip()
            if line == "readyok":
                break

    def eval_fen(self, fen: str, movetime_ms: int = 10) -> int:
        """
        Evaluates a FEN position with Stockfish and returns an integer evaluation in centipawns
        from the side to move's perspective.

        If Stockfish reports a mate score, we map it to a large centipawn value.
        """
        # Set position
        self._send(f"position fen {fen}")
        self._send("isready")
        while True:
            line = self._read_line().strip()
            if line == "readyok":
                break

        # Go command
        self._send(f"go movetime {movetime_ms}")

        eval_cp: Optional[int] = None
        while True:
            line = self._read_line().strip()
            if not line:
                continue

            if line.startswith("info") and "score" in line:
                parts = line.split()
                # find "score"
                try:
                    idx = parts.index("score")
                except ValueError:
                    continue
                kind = parts[idx + 1]
                value = parts[idx + 2]

                if kind == "cp":
                    eval_cp = int(value)
                elif kind == "mate":
                    mate_in = int(value)
                    # Map mate to a large cp. Positive for winning, negative for losing.
                    sign = 1 if mate_in > 0 else -1
                    eval_cp = sign * 100000

            elif line.startswith("bestmove"):
                # search finished
                break

        if eval_cp is None:
            # Fallback; in practice this should not happen
            eval_cp = 0

        return eval_cp

    def close(self):
        if self.proc.poll() is None:
            self._send("quit")
            self.proc.wait()


def label_position_with_stockfish(
    fen: str,
    engine: StockfishUCI,
    movetime_ms: int = 10,
) -> Dict:
    """
    Takes a FEN, queries Stockfish, and returns a dict containing:
      {
        "fen": str,
        "tensor": np.ndarray (C, 8, 8),
        "eval_cp": int
      }
    """
    tensor = fen_to_tensor(fen)
    eval_cp = engine.eval_fen(fen, movetime_ms=movetime_ms)
    return {
        "fen": fen,
        "tensor": tensor,
        "eval_cp": eval_cp,
    }


# Example high-level usage: build a dataset from a PGN file

def build_dataset_from_pgn(
    pgn_path: str,
    engine_path: str,
    max_games: int = 50_000,
    positions_per_game: int = 6,
    movetime_ms: int = 10,
):
    """
    Example driver that:
      - iterates through games in a PGN
      - samples up to `positions_per_game` FENs per game (using the opening/mid/end heuristic)
      - labels each with Stockfish
      - returns two arrays: X (tensors) and y (evals in cp)

    This is just to show how all the pieces fit together.
    """
    engine = StockfishUCI(engine_path)
    X = []
    y = []

    games_processed = 0
    with open(pgn_path, "r", encoding="utf-8") as f:
        while games_processed < max_games:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            fens = sample_fens_from_game(game)
            for fen in fens:
                label = label_position_with_stockfish(
                    fen, engine, movetime_ms=movetime_ms
                )
                X.append(label["tensor"])
                y.append(label["eval_cp"])

            games_processed += 1
            if games_processed % 1000 == 0:
                print(f"Processed {games_processed} games...")

    engine.close()

    X = np.stack(X, axis=0)  # shape: (N, C, 8, 8)
    y = np.array(y, dtype=np.float32)  # shape: (N,)

    print(f"Total positions: {len(y)}")
    return X, y


if __name__ == "__main__":
    # print(count_games_in_pgn(pgn_path))
    with open(pgn_path, "r", encoding="utf-8") as f:
        for i in range(200000):
            game = chess.pgn.read_game(f)
            if game is None:
                break
            sample_fens_from_game_to_file(game)
            if i % 1000 == 0:
                print(f"{i} FEN were loaded")