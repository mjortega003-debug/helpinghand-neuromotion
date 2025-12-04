import argparse
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# ---------- Model ----------
class BicepsMLP(nn.Module):
    def __init__(self, in_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ---------- Dataset ----------
class WindowDataset(Dataset):
    """
    Simple dataset wrapping X (N,64), y (N,2).
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.shape[0] == y.shape[0], "X and y must have same N"
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.X[idx]),
            torch.from_numpy(self.y[idx]),
        )

# ---------- Metrics ----------
def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Mean absolute error over both outputs.
    """
    with torch.no_grad():
        return torch.mean(torch.abs(pred - target)).item()

# ---------- Training / eval ----------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_mae = 0.0
    n_samples = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        batch_size = xb.size(0)
        running_loss += loss.item() * batch_size
        running_mae += mae(out, yb) * batch_size
        n_samples += batch_size

    return running_loss / n_samples, running_mae / n_samples


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    n_samples = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)

            batch_size = xb.size(0)
            running_loss += loss.item() * batch_size
            running_mae += mae(out, yb) * batch_size
            n_samples += batch_size

    return running_loss / n_samples, running_mae / n_samples

# ---------- Main ----------
def main(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    # Load data
    data = np.load(args.data_path)
    X = data["X"]
    y = data["y"]
    print(f"Loaded X{X.shape}, y{y.shape}")

    # Optionally shuffle before split (simple but OK for now)
    n = X.shape[0]
    idx = np.random.permutation(n)
    X = X[idx]
    y = y[idx]

    # Dataset and split
    dataset = WindowDataset(X, y)
    n_val = int(args.val_split * len(dataset))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model, loss, optimizer
    model = BicepsMLP(in_dim=X.shape[1]).to(device)
    criterion = nn.SmoothL1Loss()  # Huber
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    best_state = None
    patience = args.patience
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_mae = train_one_epoch(model, train_loader, optimizer, device, criterion)
        val_loss, val_mae = eval_epoch(model, val_loader, device, criterion)

        print(
            f"Epoch {epoch:03d}: "
            f"train_loss={train_loss:.4f}, train_mae={train_mae:.4f} | "
            f"val_loss={val_loss:.4f}, val_mae={val_mae:.4f}"
        )

        # Early stopping
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping.")
                break

    # Save best model
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), args.out_model)
    print(f"Saved best model to {args.out_model} (val_loss={best_val_loss:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/features_labels.npz",
                        help="Path to NPZ with X (N,64), y (N,2)")
    parser.add_argument("--out_model", type=str, default="biceps_mlp.pt",
                        help="Where to save the trained model state_dict")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Fraction of data used for validation")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if GPU is available")

    args = parser.parse_args()
    main(args)