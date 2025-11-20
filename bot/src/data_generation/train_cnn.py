
# /Users/srirenukasayanthan/Desktop/chesshacks/my-chesshacks-bot/src/data_generation
import os
import csv
import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import chess

# =========================
#  FEN -> tensor encoding
# =========================

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
               chess.ROOK, chess.QUEEN, chess.KING]


def fen_to_tensor(fen: str) -> np.ndarray:
    """
    FEN -> tensor of shape (18, 8, 8), float32.

    Channels:
      0-5  : white P, N, B, R, Q, K
      6-11 : black P, N, B, R, Q, K
      12   : side to move (1 if white to move, else 0)
      13   : white castle K
      14   : white castle Q
      15   : black castle K
      16   : black castle Q
      17   : en-passant target square, 1 at the ep square if any
    """
    board = chess.Board(fen)
    tensor = np.zeros((18, 8, 8), dtype=np.float32)

    # Piece planes
    for square, piece in board.piece_map().items():
        pt = piece.piece_type
        color = piece.color  # True=white, False=black
        type_idx = PIECE_TYPES.index(pt)
        plane = type_idx if color == chess.WHITE else 6 + type_idx

        file = chess.square_file(square)  # 0..7 (a..h)
        rank = chess.square_rank(square)  # 0..7 (1..8)
        row = 7 - rank                    # row 0 = rank 8
        col = file
        tensor[plane, row, col] = 1.0

    # Side to move
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0

    # Castling rights
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[16, :, :] = 1.0

    # En passant
    if board.ep_square is not None:
        file = chess.square_file(board.ep_square)
        rank = chess.square_rank(board.ep_square)
        row = 7 - rank
        col = file
        tensor[17, row, col] = 1.0

    return tensor


# =========================
#  Dataset
# =========================

class ChessEvalDataset(Dataset):
    """
    Reads a CSV with columns: fen, eval_cp
    and yields (tensor, target) pairs, where:
      tensor: (18, 8, 8) float32
      target: scalar float32 (normalized eval)
    """

    def __init__(self, csv_path: str):
        self.rows: List[dict] = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.rows.append(row)

        print(f"Loaded {len(self.rows)} positions from {csv_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        fen = row["fen"]
        # adjust key name if you wrote "eval" instead of "eval_cp"
        eval_cp = float(row["eval_cp"])

        x_np = fen_to_tensor(fen)           # (18, 8, 8) np.float32
        x = torch.from_numpy(x_np)         # (18, 8, 8) torch.float32

        # Normalize eval: cp -> roughly [-10,10] by dividing by 400 and clamping
        y_val = eval_cp / 400.0
        y_val = max(min(y_val, 10.0), -10.0)
        y = torch.tensor([y_val], dtype=torch.float32)  # shape (1,)

        return x, y


# =========================
#  Simple CNN model
# =========================

class ChessCNN(nn.Module):
    def __init__(self, in_channels: int = 18):
        super().__init__()
        # small but decent CNN
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Global average pooling: (B, C, 8, 8) -> (B, C)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),  # scalar eval
        )

    def forward(self, x):
        x = self.features(x)               # (B, 128, 8, 8)
        x = self.global_pool(x)            # (B, 128, 1, 1)
        x = x.view(x.size(0), -1)          # (B, 128)
        out = self.head(x)                 # (B, 1)
        return out


# =========================
#  Training loop
# =========================

def train(
    csv_path: str,
    batch_size: int = 256,
    lr: float = 1e-3,
    num_epochs: int = 50,            # ⬅ default 50 now
    val_split: float = 0.1,
    num_workers: int = 2,
    device: str = None,
    weights_path: str = "chess_cnn_eval.pt",
    resume_if_exists: bool = True,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Dataset + split
    full_dataset = ChessEvalDataset(csv_path)
    n_total = len(full_dataset)
    n_val = int(math.floor(val_split * n_total))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])
    print(f"Train positions: {n_train}, Val positions: {n_val}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    # Model, loss, optimizer
    model = ChessCNN(in_channels=18).to(device)

    # 🔁 Resume from existing weights if requested and file exists
    if resume_if_exists and os.path.exists(weights_path):
        print(f"Found existing weights at '{weights_path}'. Resuming training from them.")
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        if resume_if_exists:
            print(f"No existing weights found at '{weights_path}'. Training from scratch.")
        else:
            print("resume_if_exists=False, training from scratch.")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    base_name, ext = os.path.splitext(weights_path)

    for epoch in range(1, num_epochs + 1):
        # ---- Train ----
        model.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)  # (B,1)

            optimizer.zero_grad()
            outputs = model(inputs)       # (B,1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 50 == 0:
                print(f"Epoch {epoch} | Step {i+1}/{len(train_loader)} "
                      f"| Train Loss: {running_loss / (i+1):.4f}")

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= max(len(val_loader), 1)
        print(f"Epoch {epoch} finished. Val Loss: {val_loss:.4f}")

        # ---- Save checkpoint every 10 epochs ----
        if epoch % 10 == 0:
            ckpt_path = f"{base_name}_epoch_{epoch}{ext}"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

    # Save final model at the end
    torch.save(model.state_dict(), weights_path)
    print(f"Final model saved to {weights_path}")


# =========================
#  Main
# =========================

if __name__ == "__main__":
    # Adjust this path to your labels file
    CSV_PATH = "fen_labels.csv"

    train(
        csv_path=CSV_PATH,
        batch_size=256,
        lr=1e-3,
        num_epochs=50,       # bump if you have time
        val_split=0.1,
        num_workers=2,
    )