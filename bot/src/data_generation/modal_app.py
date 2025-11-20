import os
import sys
from pathlib import Path

import modal

# --------------------------------------------------------------------
# App + Volume setup
# --------------------------------------------------------------------

app = modal.App("chess-engine-cnn")

# Persistent volume for model weights (created lazily if missing)
volume = modal.Volume.from_name("chess-engine-cnn-models", create_if_missing=True)
MODEL_DIR = Path("/models")  # path *inside* the container/volume

# Local directory containing modal_app.py, train_cnn.py, fen_labels.csv, chess_cnn_eval.pt
LOCAL_DIR = Path(__file__).resolve().parent

# --------------------------------------------------------------------
# Image definition: Python env + our local source
# --------------------------------------------------------------------
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "numpy",
        "python-chess",
    )
    # Add the local project directory into the container at /app
    .add_local_dir(local_path=str(LOCAL_DIR), remote_path="/app")
)

# --------------------------------------------------------------------
# Training function (for the CNN in train_cnn.py)
# --------------------------------------------------------------------
@app.function(
    image=image,
    gpu="B200",   # adjust to your actual GPU type on Modal, e.g. "A10G" / "A100" / "L4" / "L40S"
    cpu="8.0",
    timeout=60 * 60 * 12,  # 12 hours; adjust as needed
    volumes={str(MODEL_DIR): volume},  # attach volume at /models
)
def train_cnn_remote(
    num_epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_split: float = 0.1,
    num_workers: int = 2,
    resume_if_exists: bool = True,
):
    """
    Run the CNN training loop defined in train_cnn.py on a GPU.

    - Reads fen_labels.csv from /app (your repo, copied into the container)
    - Saves chess_cnn_eval.pt into the Modal Volume mounted at /models
    - If chess_cnn_eval.pt already exists in the volume and resume_if_exists=True,
      training will resume from those weights.
    """
    # Make sure Python can import our project code from /app
    sys.path.insert(0, "/app")

    # Optional: print contents of /app for debugging
    print("Contents of /app in container:")
    for p in Path("/app").iterdir():
        print(" -", p.name)

    # Import your CNN trainer
    import train_cnn  # this is your train_cnn.py

    # Ensure the model directory exists in the volume
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("Model volume directory:", MODEL_DIR)

    # We want the *output* file chess_cnn_eval.pt to land in the volume.
    # train_cnn.train() saves to weights_path in the CURRENT WORKING DIRECTORY,
    # so we temporarily chdir into /models.
    os.chdir(MODEL_DIR)

    csv_path = Path("/app") / "fen_labels.csv"
    print("Using CSV file:", csv_path)

    # Call your training entrypoint
    train_cnn.train(
        csv_path=str(csv_path),
        batch_size=batch_size,
        lr=lr,
        num_epochs=num_epochs,
        val_split=val_split,
        num_workers=num_workers,
        device=None,  # let train_cnn pick cuda/cpu
        weights_path="chess_cnn_eval.pt",   # lives in /models inside the volume
        resume_if_exists=resume_if_exists,  # <- NEW: resume if file already exists
    )

    # At this point, /models/chess_cnn_eval.pt is the trained/updated model
    print("Finished training. Model saved to:", MODEL_DIR / "chess_cnn_eval.pt")


# --------------------------------------------------------------------
# Local entrypoint: lets you call this from your terminal
# --------------------------------------------------------------------
@app.local_entrypoint()
def main(
    num_epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_split: float = 0.1,
    num_workers: int = 2,
    resume_if_exists: bool = True,
):
    """
    Example usage from your terminal:

        modal run modal_app.py --num-epochs 10 --batch-size 256 --lr 0.001

    By default, if /models/chess_cnn_eval.pt already exists in the Modal Volume,
    training will resume from it. Pass --resume-if-exists False to force training
    from scratch.
    """
    print("Submitting CNN training job to Modal…")
    train_cnn_remote.remote(
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        val_split=val_split,
        num_workers=num_workers,
        resume_if_exists=resume_if_exists,
    )