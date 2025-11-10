from src.utils.mock_data_generator import MockDataGenerator
from src.core.signal_cleaner import SignalCleaner
from src.core.intent_detector import IntentDetector
from src.core.command_builder import CommandBuilder
from src.core.session_recorder import SessionRecorder

class UltraCortexSimulation:
    # Central class that simulates a single EEG session.
    # Handles the workflow:
    # 1. Load/generate mock EEG
    # 2. Clean & extract features
    # 3. Detect intent
    # 4. Build command
    # 5. Log results

    def __init__(self):
        self.data_generator = MockDataGenerator()
        self.signal_cleaner = SignalCleaner()
        self.intent_detector = IntentDetector()
        self.command_builder = CommandBuilder()
        self.session_recorder = SessionRecorder()

    def start_simulation(self):
        # 1. Generate mock EEG data
        eeg_data = self.data_generator.generate_mock_data()
        # 2. Clean & extract features
        features = self.signal_cleaner.preprocess_and_extract(eeg_data)
        # 3. Predict intent
        intent = self.intent_detector.classify(features)
        # 4. Convert intent to command
        command = self.command_builder.generate_command(intent)
        # 5. Record the session
        self.session_recorder.log(intent, command)
        print(f"Intent: {intent} | Command: {command}")
