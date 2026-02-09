import csv
import os
import time
from pylsl import resolve_streams

class DataProcessor:
    """
    Discovers LSL streams (EKG, EMG, CV), and initializes a CSV output file.
    Later: will pull samples + write continuous rows.
    """

    def run(self):
        print("DataProcessor initialized")
        print("Listening for LSL streams / preparing CSV pipeline")
    
    def __init__(self, output_file="logs/neuromotion_data.csv"):
        self.output_file = output_file
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

    def init_csv(self):
        # Only write headers if file doesn't exist or is empty
        needs_header = (not os.path.exists(self.output_file)) or os.path.getsize(self.output_file) == 0
        if needs_header:
            with open(self.output_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "ekg_value", "emg_value", "cv_label"])
        print("CSV pipeline initialized:", self.output_file)

    def find_streams(self):
        print("Looking for LSL streams...")
        streams = resolve_streams()
        if not streams:
            print("No LSL streams found.")
            return []
        for s in streams:
            print("---")
            print("Name:", s.name())
            print("Type:", s.type())
            print("Channels:", s.channel_count())
        return streams

    def run(self):
        self.init_csv()
        self.find_streams()
        # placeholder for continuous processing later
        # (ex: while True: pull_sample from inlets and write rows)
        time.sleep(0.1)

        if __name__ == "__main__":
         DataProcessor().run()

