from src.utils.mock_data_generator import MockDataGenerator
from src.core.signal_cleaner import SignalCleaner
from src.core.intent_detector import IntentDetector
from src.core.command_builder import CommandBuilder
from src.core.session_recorder import SessionRecorder

# NEW IMPORT
from src.hardware.headset_connector import HeadsetConnector
from brainflow.board_shim import BoardIds

import yaml
import time
import numpy as np

class UltraCortexSimulation:
    """
    Main orchestrator for EEG → processing → intent → commands
    Automatically switches between:
    - synthetic EEG (mock)
    - real EEG via BrainFlow
    """

    def __init__(self, config_path="configs/app_config.yaml"):
        self.config = yaml.safe_load(open(config_path, "r"))

        self.mode = self.config["input_source"]  # "mock" or "ultracortex"
        print(f"[UltraCortexSimulation] Input source: {self.mode}")

        # Shared modules
        self.signal_cleaner = SignalCleaner()
        self.intent_detector = IntentDetector()
        self.command_builder = CommandBuilder()
        self.session_recorder = SessionRecorder()

        # Set up either mock or hardware
        if self.mode == "mock":
            self.data_source = MockDataGenerator(
                channels=self.config["mock"]["channels"],
                duration=self.config["mock"]["duration"],
                sample_rate=self.config["mock"]["sample_rate"]
            )
        else:
            # UltraCortex typically uses CYTON or GANGLION
            board_id = BoardIds.CYTON_BOARD.value
            serial_port = self.config["ultracortex"]["serial_port"]
            self.data_source = HeadsetConnector(board_id, serial_port)

    def start_simulation(self):
        """
        Start a continuous simulation loop.
        - If mode == "mock": run a single mock session (unchanged behavior).
        - If mode == "ultracortex": connect, start stream, wait for buffer fill,
        then continuously poll data and process it until user stops with Ctrl+C.
        """
        print("[UltraCortexSimulation] Starting simulation...")

        # Handle hardware connection if in ultracortex mode
        if self.mode == "ultracortex":
            print("[UltraCortexSimulation] Trying to connect to UltraCortex...")
            success = self.data_source.connect()  # data_source is HeadsetConnector

            if not success:
                print("[UltraCortexSimulation] UltraCortex connection failed.")
                print("[UltraCortexSimulation] Falling back to mock EEG mode...\n")
                self.mode = "mock"
                # Re-initialize the data source as MockDataGenerator
                self.data_source = MockDataGenerator(
                    channels=self.config["mock"]["channels"],
                    duration=self.config["mock"]["duration"],
                    sample_rate=self.config["mock"]["sample_rate"]
                )
            else:
                # Connection successful, start the stream
                self.data_source.start_stream()
                # Give BrainFlow some time to fill internal buffers (adjustable)
                buffer_warmup = self.config.get("ultracortex", {}).get("buffer_warmup", 1.0)
                print(f"[UltraCortexSimulation] Waiting {buffer_warmup:.2f}s for buffer warmup...")
                time.sleep(buffer_warmup)

        # If mock mode, keep the old single-shot behavior (useful for tests)
        if self.mode == "mock":
            eeg_data = self.data_source.generate_mock_data()

            # Safety check
            if eeg_data is None:
                print("[UltraCortexSimulation] No mock EEG data retrieved.")
                return

            features = self.signal_cleaner.preprocess_and_extract(eeg_data)
            intent = self.intent_detector.classify(features)
            command = self.command_builder.generate_command(intent)
            self.session_recorder.log(intent, command)
            print(f"[UltraCortexSimulation] Intent: {intent} | Command: {command}")
            return

        # From here: real-time ultracortex streaming loop
        polling_interval = self.config.get("ultracortex", {}).get("poll_interval", 0.1)  # seconds
        num_samples = self.config.get("ultracortex", {}).get("pull_samples", 50)

        print("[UltraCortexSimulation] Entering real-time loop (Ctrl+C to stop)...")
        try:
            while True:
                eeg_data = self.data_source.get_data(num_samples=num_samples)

                # If no data yet, skip and wait
                if eeg_data is None:
                    time.sleep(polling_interval)
                    continue

                # Convert to numpy array if not already
                eeg_arr = np.asarray(eeg_data)
                # BrainFlow returns matrix with shape (channels, samples) or similar.
                if eeg_arr.size == 0:
                    # empty buffer — wait a bit and continue
                    time.sleep(polling_interval)
                    continue

                # Basic sanity check: replace NaNs or infinities if present
                if np.isnan(eeg_arr).any() or np.isinf(eeg_arr).any():
                    print("[UltraCortexSimulation] Warning: NaN/Inf in EEG chunk — skipping")
                    time.sleep(polling_interval)
                    continue

                # Save raw EEG block every iteration
                self.data_source.save_data(num_samples)

                # PROCESS EEG → features → intent → command
                features = self.signal_cleaner.preprocess_and_extract(eeg_arr)
                intent = self.intent_detector.classify(features)
                command = self.command_builder.generate_command(intent)

                # Log results
                self.session_recorder.log(intent, command)

                # Output
                print(f"[UltraCortexSimulation] Intent: {intent} | Command: {command}")

                # Small sleep to control polling rate
                time.sleep(polling_interval)

        except KeyboardInterrupt:
            print("\n[UltraCortexSimulation] KeyboardInterrupt received — shutting down...")

        finally:
            # Clean up hardware if real device
            if self.mode == "ultracortex":
                try:
                    self.data_source.stop_stream()
                except Exception:
                    pass
                try:
                    self.data_source.disconnect()
                except Exception:
                    pass

            print("[UltraCortexSimulation] Simulation stopped.")