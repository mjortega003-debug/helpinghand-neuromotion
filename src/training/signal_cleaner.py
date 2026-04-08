import numpy as np
from scipy.signal import butter, lfilter, iirnotch

class SignalCleaner:
    """
    Filters biosignals (EEG/EMG) and extracts time-window features.
    """
    def __init__(self, fs=250):
        self.fs = fs
        # 60Hz Notch filter for North American powerline noise (use 50 for Europe/Asia)
        self.notch_b, self.notch_a = iirnotch(w0=60.0, Q=30.0, fs=self.fs)
        # Bandpass filter (1Hz to 40Hz)
        nyq = 0.5 * self.fs
        self.bp_b, self.bp_a = butter(N=5, Wn=[1.0/nyq, 40.0/nyq], btype='band')

    def preprocess_and_extract(self, data):
        """
        Takes raw window data of shape (channels, samples).
        Returns a feature vector (RMS) of shape (channels,).
        """
        # 1. Apply 60Hz Notch Filter to kill electrical hum
        notched_data = lfilter(self.notch_b, self.notch_a, data, axis=1)
        
        # 2. Apply Bandpass Filter to isolate motor frequencies
        filtered_data = lfilter(self.bp_b, self.bp_a, notched_data, axis=1)
        
        # 3. Extract Feature: Root Mean Square (RMS)
        # RMS is much better than mean/std for muscle and neural power
        rms_features = np.sqrt(np.mean(filtered_data**2, axis=1))
        
        return rms_features