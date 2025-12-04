import numpy as np

"""""
X: 64 features per sample (fake EEG windows)
y: 2 outputs per sample (left, right bicep intensity)
"""

def make_fake_dataset(N=2000):
    # Simulate 64 features
    X = np.random.randn(N, 64).astype(np.float32)

    # Make fake intensities:
    # right arm intensity increases when first 10 features rise
    # left arm intensity increases when next 10 features rise
    left_raw = X[:, :10].mean(axis=1)
    right_raw = X[:, 10:20].mean(axis=1)

    # normalize to [0,1]
    left = (left_raw - left_raw.min()) / (left_raw.max() - left_raw.min())
    right = (right_raw - right_raw.min()) / (right_raw.max() - right_raw.min())

    y = np.stack([left, right], axis=1).astype(np.float32)

    np.savez("fake_data.npz", X=X, y=y)
    print("Created fake_data.npz with shapes:", X.shape, y.shape)


if __name__ == "__main__":
    make_fake_dataset()