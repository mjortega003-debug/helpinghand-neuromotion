from brainflow.board_shim import BoardShim, BrainFlowInputParams, BrainFlowError
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations

class HeadsetConnector:
    """
    Handles connection, data streaming, and EEG retrieval from a BrainFlow-supported board.
    """

    def __init__(self, board_id, serial_port=None):
        self.board_id = board_id
        self.serial_port = serial_port

        self.params = BrainFlowInputParams()
        self.params.serial_port = serial_port

        self.board = BoardShim(board_id, self.params)
        self.streaming = False

    def connect(self):
        try:
            BoardShim.enable_dev_board_logger()
            self.board.prepare_session()
            print(f"[HeadsetConnector] Session prepared for board {self.board_id}")
        except BrainFlowError as e:
            print(f"[HeadsetConnector] ERROR preparing session: {e}")

    def start_stream(self, buffer_size=45000):
        try:
            self.board.start_stream(buffer_size)
            self.streaming = True
            print("[HeadsetConnector] Stream started.")
        except BrainFlowError as e:
            print(f"[HeadsetConnector] ERROR starting stream: {e}")

    def get_data(self, num_samples=250):
        if not self.streaming:
            print("[HeadsetConnector] WARNING: get_data() called, but stream is not running.")
            return None

        data = self.board.get_board_data(num_samples)
        return data

    def stop_stream(self):
        try:
            if self.streaming:
                self.board.stop_stream()
                print("[HeadsetConnector] Stream stopped.")
        except BrainFlowError:
            pass

    def disconnect(self):
        try:
            self.board.release_session()
            print("[HeadsetConnector] Session released.")
        except BrainFlowError:
            pass