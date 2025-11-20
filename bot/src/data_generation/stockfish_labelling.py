
# /Users/srirenukasayanthan/Desktop/chesshacks/my-chesshacks-bot/src/data_generation

import os
import csv
import subprocess
from typing import Optional, Tuple
import multiprocessing as mp


# ------------ Stockfish UCI Wrapper ------------

class StockfishUCI:
    """
    Minimal UCI wrapper for a local Stockfish binary.
    """

    def __init__(self, engine_path: str = "/Users/srirenukasayanthan/Downloads/stockfish/stockfish-macos-m1-apple-silicon"):
        # If a non-default path is given, sanity check it
        if engine_path != "/Users/srirenukasayanthan/Downloads/stockfish/stockfish-macos-m1-apple-silicon" and not os.path.exists(engine_path):
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

        # Optional: force 1 thread per engine for better parallel scaling
        self._send("setoption name Threads value 1")
        self._send("isready")
        while True:
            line = self._read_line().strip()
            if line == "readyok":
                break

    def eval_fen(self, fen: str, movetime_ms: int = 10) -> int:
        """
        Evaluate a FEN. Returns integer centipawns from the side to move perspective.
        If a mate score is reported, map it to ±100000 cp.
        """
        # Set position
        self._send(f"position fen {fen}")
        self._send("isready")
        while True:
            line = self._read_line().strip()
            if line == "readyok":
                break

        # Start search
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

                kind = parts[idx + 1]  # "cp" or "mate"
                value = parts[idx + 2]

                if kind == "cp":
                    eval_cp = int(value)
                elif kind == "mate":
                    mate_in = int(value)
                    sign = 1 if mate_in > 0 else -1
                    eval_cp = sign * 100000

            elif line.startswith("bestmove"):
                # search finished
                break

        if eval_cp is None:
            eval_cp = 0  # fallback, shouldn't happen often

        return eval_cp

    def close(self):
        if self.proc.poll() is None:
            self._send("quit")
            self.proc.wait()


# ------------ Multiprocessing setup ------------

# Global variables inside each worker process
ENGINE = None
MOVETIME_MS = None


def worker_init(engine_path: str, movetime_ms: int):
    """
    Initializer for each worker process.
    Creates its own Stockfish instance and stores it in a global variable.
    """
    global ENGINE, MOVETIME_MS
    MOVETIME_MS = movetime_ms
    ENGINE = StockfishUCI(engine_path)


def worker_eval_fen(fen: str) -> Optional[Tuple[str, int]]:
    """
    Worker function: takes a FEN string, returns (fen, eval_cp).
    Returns None for blank lines.
    """
    fen = fen.strip()
    if not fen:
        return None

    global ENGINE, MOVETIME_MS
    eval_cp = ENGINE.eval_fen(fen, movetime_ms=MOVETIME_MS)
    return fen, eval_cp


# ------------ Main driver ------------

def label_fen_file(
    fen_file: str,
    output_csv: str,
    engine_path: str = "/Users/srirenukasayanthan/Downloads/stockfish/stockfish-macos-m1-apple-silicon",
    num_workers: int = 4,
    movetime_ms: int = 10,
    max_positions: Optional[int] = None,
):
    """
    Reads FENs line-by-line from `fen_file`, labels each with Stockfish, and writes:
        fen, eval_cp
    to `output_csv`.

    Args:
        fen_file: path to file with one FEN per line.
        output_csv: path to output CSV.
        engine_path: path to Stockfish binary.
        num_workers: number of parallel worker processes (and Stockfish instances).
        movetime_ms: movetime per position in milliseconds.
        max_positions: optional cap on how many positions to label.
    """
    # Count lines if you want a progress estimate (optional)
    if max_positions is None:
        with open(fen_file, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)
    else:
        total_lines = max_positions

    print(f"Labeling up to {total_lines} positions with {num_workers} workers...")

    with open(fen_file, "r", encoding="utf-8") as f_in, \
         open(output_csv, "w", newline="", encoding="utf-8") as f_out:

        writer = csv.writer(f_out)
        writer.writerow(["fen", "eval_cp"])

        # Create a generator over the FEN lines, optionally truncated
        def fen_iter():
            count = 0
            for line in f_in:
                if max_positions is not None and count >= max_positions:
                    break
                yield line
                count += 1

        # Pool of workers, each with its own Stockfish
        with mp.Pool(
            processes=num_workers,
            initializer=worker_init,
            initargs=(engine_path, movetime_ms)
        ) as pool:
            # chunksize helps reduce IPC overhead
            for idx, result in enumerate(pool.imap_unordered(worker_eval_fen, fen_iter(), chunksize=64), start=1):
                if result is None:
                    continue
                fen, eval_cp = result
                writer.writerow([fen, eval_cp])

                if idx % 1000 == 0:
                    print(f"Labeled {idx} / {total_lines} positions...")

    print(f"Done. Labeled positions written to {output_csv}")


if __name__ == "__main__":
    # Example usage:
    # Adjust paths & params for your machine.
    fen_input_file = "all_positions.fen"      # your 479,950-line FEN file
    output_labels = "fen_labels.csv"
    stockfish_bin = "/Users/srirenukasayanthan/Downloads/stockfish/stockfish-macos-m1-apple-silicon"              # or absolute path
    workers = 4                              # set to number of CPU cores
    movetime = 10                            # ms per position

    label_fen_file(
        fen_file=fen_input_file,
        output_csv=output_labels,
        engine_path=stockfish_bin,
        num_workers=workers,
        movetime_ms=movetime,
        max_positions=None,   # or e.g. 100000 to test
    )