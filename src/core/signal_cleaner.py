import numpy as np
from scipy.signal import butter, lfilter

class SignalCleaner:
# Handles filtering and feature extraction from EEG data.
    
    def preprocess_and_extract(self, data):
        # Applies a bandpass filter and computes simple features.
        # Returns: feature vector (np.ndarray)
        filtered = self.bandpass_filter(data, lowcut=1, highcut=40, fs=250)
        features = self.extract_features(filtered)
        return features

    def bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low, high = lowcut / nyq, highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, data)

    def extract_features(self, data):
        # Example: compute mean and std across channels.
        mean = np.mean(data, axis=1)
        std = np.std(data, axis=1)
        return np.concatenate([mean, std])
