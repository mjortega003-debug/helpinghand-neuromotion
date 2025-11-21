from src.utils.mock_data_generator import MockDataGenerator
from src.core.signal_cleaner import SignalCleaner
from src.core.intent_detector import IntentDetector
from src.core.command_builder import CommandBuilder
from src.core.session_recorder import SessionRecorder

# NEW IMPORT
from src.hardware.headset_connector import HeadsetConnector
from brainflow.board_shim import BoardIds

import yaml

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

        # Retrieve EEG
        if self.mode == "mock":
            eeg_data = self.data_source.generate_mock_data()
        else:
            eeg_data = self.data_source.get_data(
                num_samples=self.config["ultracortex"]["pull_samples"]
            )

        # Safety check
        if eeg_data is None:
            print("[UltraCortexSimulation] No EEG data retrieved.")
            return

        # Cleaning + feature extraction
        features = self.signal_cleaner.preprocess_and_extract(eeg_data)
    
        # Intent classification
        intent = self.intent_detector.classify(features)

        # Command generation
        command = self.command_builder.generate_command(intent)

        # Logging
        self.session_recorder.log(intent, command)

        print(f"[UltraCortexSimulation] Intent: {intent} | Command: {command}")

        # Clean up hardware if real device
        if self.mode == "ultracortex":
            self.data_source.stop_stream()
            self.data_source.disconnect()