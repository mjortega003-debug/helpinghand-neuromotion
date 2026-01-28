import csv
from pylsl import StreamInlet, resolve_byprop

# 1. Setup the CSV file for synchronized data
filename = "synced_bci_gestures.csv"
# Assuming an 8-channel OpenBCI setup + 1 for the Gesture
fields = ['lsl_timestamp', 'ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8', 'gesture']

with open(filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(fields)

# 2. Find the Streams
print("Searching for LSL streams...")
# Look for the MediaPipe Gesture stream
gesture_streams = resolve_byprop('type', 'Markers', timeout=5)
# Look for the OpenBCI GUI stream (Usually type 'EEG' or 'EXG')
bci_streams = resolve_byprop('type', 'EEG', timeout=5)

if not gesture_streams:
    print("no MediaPipe is broadcasting.")
    exit()
if not bci_streams:
    print("no OpenBCI GUI broadcasting.")
    exit()

# Create inlets
gesture_inlet = StreamInlet(gesture_streams[0])
bci_inlet = StreamInlet(bci_streams[0])

print(f"Connected to both! Logging to {filename}...")
current_gesture = "None"  # Keep track of the last detected gesture

try:
    while True:
        # 3. Pull Gesture (Non-blocking)
        # We only update the 'current_gesture' when a new one arrives
        g_sample, g_timestamp = gesture_inlet.pull_sample(timeout=0.0)
        if g_sample:
            current_gesture = g_sample[0]

        # 4. Pull BCI Data (Blocking, as this is our primary high-speed data)
        b_sample, b_timestamp = bci_inlet.pull_sample()

        if b_sample:
            # Combine BCI channels with the latest gesture label and the BCI timestamp
            row = [b_timestamp] + b_sample + [current_gesture]

            with open(filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

            # Print occasionally so the console isn't a blur
            if int(b_timestamp) % 10 == 0:
                print(f"Synced Sample: {current_gesture} | Signal: {b_sample[0]:.2f}uV")

except KeyboardInterrupt:
    print("\nData collection complete.")