class EMASmoother:
    def __init__(self, alpha: float = 0.6):
        self.alpha = alpha
        self.state = None

    def step(self, x: np.ndarray) -> np.ndarray:
        # x: (2,) array of [left, right] intensities
        if self.state is None:
            self.state = x.astype(float)
        else:
            self.state = self.alpha * x + (1.0 - self.alpha) * self.state
        return self.state
