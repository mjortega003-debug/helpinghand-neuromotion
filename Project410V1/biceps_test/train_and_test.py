import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------
# MODEL
# --------------------------------------------------------
class BicepsMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()     # outputs always in [0,1]
        )

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------
# TRAINING
# --------------------------------------------------------
def train_model():
    data = np.load("fake_data.npz")
    X = torch.tensor(data["X"])
    y = torch.tensor(data["y"])

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = BicepsMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    print("Training on fake dataset...")

    for epoch in range(10):
        running_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/10   Loss: {running_loss:.4f}")

    torch.save(model.state_dict(), "biceps_model.pt")
    print("\nSaved model → biceps_model.pt")

    return model


# --------------------------------------------------------
# TEST
# --------------------------------------------------------
def test_model(model):
    print("\nRunning a realtime-style test...")

    # Make a single fake EEG window (64 features)
    fake_window = torch.randn(1, 64)

    # Run inference
    output = model(fake_window).detach().numpy()[0]

    print(f"Input window → 64 fake EEG features")
    print(f"Model output → Left: {output[0]:.3f}, Right: {output[1]:.3f}")
    print("\n(These values will be random, but within [0,1])\n")


if __name__ == "__main__":
    model = train_model()
    test_model(model)
