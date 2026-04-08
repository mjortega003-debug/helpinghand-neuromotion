import time
import numpy as np
from pylsl import resolve_streams, StreamInlet
from signal_cleaner import SignalCleaner
from intent_detector import IntentDetector

class LiveInference:
    def __init__(self, window_size=50):
        # 50 samples = 200ms of data at 250Hz
        self.window_size = window_size 
        self.cleaner = SignalCleaner(fs=250)
        self.detector = IntentDetector()
        
        self.inlets = {}
        self.eeg_buffer = []
        self.emg_buffer = []

    def connect_streams(self):
        print("[LiveInference] Searching for OpenBCI LSL streams...")
        streams = resolve_streams(wait_time=3.0)
        
        for s in streams:
            name = s.name().upper()
            if "EEG" in name: self.inlets['eeg'] = StreamInlet(s)
            elif "EMG" in name: self.inlets['emg'] = StreamInlet(s)

        if 'eeg' not in self.inlets or 'emg' not in self.inlets:
            print("[LiveInference] ERROR: Missing EEG or EMG stream. Is the OpenBCI GUI running?")
            return False
            
        print("[LiveInference] Streams connected! Ready for inference.")
        return True

    def run(self):
        if not self.connect_streams():
            return

        print("\n" + "="*50)
        print("LIVE BRAINWAVE DECODING ACTIVE")
        print("No camera connected. The AI is reading your signals.")
        print("Perform your gestures. Press Ctrl+C to stop.")
        print("="*50 + "\n")

        try:
            last_prediction = "idle"
            last_print_time = time.time()

            while True:
                # 1. Pull the live data
                sample_eeg, _ = self.inlets['eeg'].pull_sample(timeout=0.01) if 'eeg' in self.inlets else (None, None)
                sample_emg, _ = self.inlets['emg'].pull_sample(timeout=0.0) if 'emg' in self.inlets else (None, None)

                if sample_eeg and sample_emg:
                    # Keep only the first 8 channels (ignoring extra OpenBCI metadata)
                    self.eeg_buffer.append(sample_eeg[:8])
                    self.emg_buffer.append(sample_emg[:8])

                    # 2. Maintain the 200ms sliding window
                    if len(self.eeg_buffer) > self.window_size:
                        self.eeg_buffer.pop(0)
                        self.emg_buffer.pop(0)

                    # 3. Predict once the buffer is full
                    if len(self.eeg_buffer) == self.window_size:
                        
                        # Combine lists into a single window of shape (50, 16)
                        raw_window = []
                        for i in range(self.window_size):
                            raw_window.append(self.emg_buffer[i] + self.eeg_buffer[i])
                        
                        # Transpose to (16 channels, 50 samples) for the SignalCleaner
                        raw_window_np = np.array(raw_window).T 

                        # Clean the signal (Notch + Bandpass) and extract RMS
                        features = self.cleaner.preprocess_and_extract(raw_window_np)

                        # Ask the PPO model what gesture this is
                        prediction = self.detector.classify(features)

                        # Print logic: Only print if the gesture changes, or every 0.5s to avoid terminal spam
                        current_time = time.time()
                        if prediction != last_prediction or (current_time - last_print_time) > 0.5:
                            print(f">>> DETECTED INTENT: {prediction.upper()} <<<")
                            last_prediction = prediction
                            last_print_time = current_time

        except KeyboardInterrupt:
            print("\n[LiveInference] Shutting down. Great test!")

if __name__ == "__main__":
    LiveInference().run()