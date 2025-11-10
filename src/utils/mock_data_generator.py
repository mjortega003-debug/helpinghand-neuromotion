import numpy as np

class MockDataGenerator:
    # Simulates EEG readings.
    # For now, returns random data shaped like real EEG signals.
    # Later, you can model actual wave patterns or load CSVs.

    def __init__(self, channels=8, duration=5, sample_rate=250):
        self.channels = channels
        self.duration = duration
        self.sample_rate = sample_rate

    def generate_mock_data(self):
        # Returns: np.ndarray of shape (channels, samples)
        samples = self.duration * self.sample_rate
        data = np.random.randn(self.channels, samples)
        return data
