import time
import numpy as np
from pylsl import resolve_streams, StreamInlet
from signal_cleaner import SignalCleaner
from intent_detector import IntentDetector

# Path to the flat file used as an IPC bridge — an external process (e.g. a game/robot)
# polls this file to read the latest predicted gesture command.
FILE_PATH = "E:/pose_command"

# LiveInference: Pulls real-time EEG + EMG from OpenBCI, cleans the signal,
# and writes the predicted gesture intent to a file for downstream consumption.
class LiveInference:
    def __init__(self, window_size=50):
        self.window_size = window_size  # Number of samples per inference window (~200ms at 250Hz)
        self.cleaner = SignalCleaner(fs=250)  # fs must match the OpenBCI sample rate
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

    def write_pose(self, pose):
        # Overwrites the file each time — the consumer should handle stale reads gracefully
        try:
            with open(FILE_PATH, "w") as f:
                f.write(pose)
        except OSError as e:
            print(f"[LiveInference] WARNING: Could not write to {FILE_PATH}: {e}")

    def run(self):
        if not self.connect_streams():
            return

        print("\n" + "="*50)
        print("LIVE BRAINWAVE DECODING ACTIVE")
        print("Perform your gestures. Press Ctrl+C to stop.")
        print("="*50 + "\n")

        try:
            last_prediction = "idle"
            last_print_time = time.time()
            last_write_time = time.time()

            # Initialize the pose file so the consumer never reads an empty state
            self.write_pose("idle")

            while True:
                # EEG is blocking (0.01s timeout) to pace the loop; EMG is non-blocking
                sample_eeg, _ = self.inlets['eeg'].pull_sample(timeout=0.01) if 'eeg' in self.inlets else (None, None)
                sample_emg, _ = self.inlets['emg'].pull_sample(timeout=0.0) if 'emg' in self.inlets else (None, None)

                if sample_eeg and sample_emg:
                    # Only use the first 8 channels from each modality
                    self.eeg_buffer.append(sample_eeg[:8])
                    self.emg_buffer.append(sample_emg[:8])

                    # Sliding window — drop the oldest sample once at capacity
                    if len(self.eeg_buffer) > self.window_size:
                        self.eeg_buffer.pop(0)
                        self.emg_buffer.pop(0)

                    if len(self.eeg_buffer) == self.window_size:
                        # Interleave EMG + EEG per timestep → shape (window_size, 16), then transpose to (16, window_size)
                        raw_window = []
                        for i in range(self.window_size):
                            raw_window.append(self.emg_buffer[i] + self.eeg_buffer[i])
                        
                        raw_window_np = np.array(raw_window).T
                        features = self.cleaner.preprocess_and_extract(raw_window_np)
                        prediction = self.detector.classify(features)

                        current_time = time.time()

                        # Print on change or every 0.5s to avoid console spam during sustained gestures
                        if prediction != last_prediction or (current_time - last_print_time) > 0.5:
                            print(f">>> DETECTED INTENT: {prediction.upper()} <<<")
                            last_prediction = prediction
                            last_print_time = current_time

                        # Throttle file writes to 1Hz — avoids I/O overhead on every inference tick
                        if (current_time - last_write_time) >= 1.0:
                            self.write_pose(prediction)
                            last_write_time = current_time

        except KeyboardInterrupt:
            print("\n[LiveInference] Shutting down.")

if __name__ == "__main__":
    LiveInference().run()