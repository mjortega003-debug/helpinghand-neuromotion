import csv
import os
import time
import subprocess
import json
from pylsl import resolve_streams, StreamInlet

class DataProcessor:
    def __init__(self, output_file="logs/neuromotion_data.csv", camera_script=r"src\core\camera_gesture_lsl.py"):
        self.output_file = output_file
        self.camera_script = camera_script
        self.camera_process = None
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        self.inlets = {}
        self.num_channels = 8 

    def start_camera_feed(self):
        """Launches the camera gesture script for automated labeling."""
        print(f"[DataProcessor] Launching camera labeling script: {self.camera_script}...")
        self.camera_process = subprocess.Popen(["python", self.camera_script])
        print("[DataProcessor] Waiting 5s for LSL stability...")
        time.sleep(5)

    def init_csv(self):
        """Initializes the CSV with 82 columns (TS + 16 Biosignals + 3 Meta + 63 Landmarks)."""
        needs_header = (not os.path.exists(self.output_file)) or os.path.getsize(self.output_file) == 0
        if needs_header:
            with open(self.output_file, "w", newline="") as f:
                writer = csv.writer(f)
                headers = ["timestamp"]
                headers += [f"emg_ch{i+1}" for i in range(self.num_channels)]
                headers += [f"eeg_ch{i+1}" for i in range(self.num_channels)]
                headers += ["cv_label", "cv_score", "cv_handedness"]
                
                # Add 63 columns for the 21 3D landmarks (x, y, z)
                for i in range(21):
                    headers.extend([f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z"])
                    
                writer.writerow(headers)
        print(f"[DataProcessor] CSV ready for 16-channel headset + 63-point tracking data.")

    def find_and_initialize_inlets(self):
        """Locates the 3 LSL streams (EEG, EMG, and CV Label)."""
        print("[DataProcessor] Searching for UltraCortex EEG/EMG and Camera streams...")
        streams = resolve_streams(wait_time=3.0)
        
        for s in streams:
            name = s.name().upper()
            if "EMG" in name:
                self.inlets['emg'] = StreamInlet(s)
            elif "EEG" in name:
                self.inlets['eeg'] = StreamInlet(s)
            elif "CV_STREAM" in name:
                self.inlets['cv'] = StreamInlet(s)

        if 'eeg' not in self.inlets or 'emg' not in self.inlets:
            print("[DataProcessor] ERROR: Missing headset streams. Check OpenBCI GUI LSL settings.")
            return False
        
        return True

    def run(self):
        self.init_csv()
        self.start_camera_feed()
        
        if not self.find_and_initialize_inlets():
            self.cleanup()
            return

        print("\n" + "="*50)
        print("RECORDING ACTIVE: 16 HEADSET CHANNELS + RICH CAMERA DATA")
        print("Focus on your 7 positions. Press Ctrl+C to stop.")
        print("="*50 + "\n")
        
        try:
            with open(self.output_file, "a", newline="") as f:
                writer = csv.writer(f)
                last_emg = [0.0] * self.num_channels
                
                # Default empty state for the 66-item camera array
                #[label, score, handedness, lm_0_x ... lm_20_z]
                last_cv_data = ["none", 0.0, "unknown"] + [0.0] * 63
                
                while True:
                    #Pull EEG
                    sample_eeg, ts_eeg = self.inlets['eeg'].pull_sample(timeout=0.01) if 'eeg' in self.inlets else (None, None)
                    
                    #Update EMG
                    if 'emg' in self.inlets:
                        sample_emg, _ = self.inlets['emg'].pull_sample(timeout=0.0)
                        if sample_emg:
                            last_emg = sample_emg[:self.num_channels]
                    
                    #Parse JSON CV Payload
                    if 'cv' in self.inlets:
                        sample_cv, _ = self.inlets['cv'].pull_sample(timeout=0.0)
                        if sample_cv:
                            try:
                                payload = json.loads(sample_cv[0])
                                last_cv_data = [
                                    payload.get("label", "none"),
                                    payload.get("score", 0.0),
                                    payload.get("handedness", "unknown")
                                ] + payload.get("landmarks", [0.0] * 63)
                            except json.JSONDecodeError:
                                # If the JSON is corrupted in transit, ignores the sample and keeps the last known state
                                pass

                    #Write synced 82-column row
                    if sample_eeg:
                        eeg_data = sample_eeg[:self.num_channels]
                        row = [ts_eeg] + last_emg + eeg_data + last_cv_data
                        writer.writerow(row)

        except KeyboardInterrupt:
            print("\n[DataProcessor] Session ended by user.")
        finally:
            self.cleanup()

    def cleanup(self):
        if self.camera_process:
            self.camera_process.terminate()
            self.camera_process.wait()
        print("[DataProcessor] Cleanup complete. Your rich training data is saved.")

if __name__ == "__main__":
    DataProcessor().run()