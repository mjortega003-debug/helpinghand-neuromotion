from brainflow.board_shim import BoardShim, BrainFlowInputParams, BrainFlowError
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
import numpy as np
import serial.tools.list_ports

class HeadsetConnector:
    """
    Handles connection, data streaming, and EEG retrieval from a BrainFlow-supported board.
    """

    def __init__(self, board_id, serial_port=None):
        self.board_id = board_id
        self.streaming = False

        # Auto-detect port if requested
        if serial_port == "auto" or serial_port is None:
            serial_port = self.auto_detect_ultracortex_port()
            if serial_port is None:
                raise RuntimeError("[HeadsetConnector] Could not auto-detect UltraCortex port.")
            print(f"[HeadsetConnector] Auto-detected UltraCortex at: {serial_port}")

        self.params = BrainFlowInputParams()
        self.params.serial_port = serial_port

        # Create the BrainFlow board
        self.board = BoardShim(board_id, self.params)

    @staticmethod
    def auto_detect_ultracortex_port():
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            # UltraCortex / Cyton dongle VID/PID
            if p.vid == 0x0403 and p.pid == 0x6015:
                return p.device
        return None

    def connect(self) -> bool:
        try:
            BoardShim.enable_dev_board_logger()
            self.board.prepare_session()
            self.connection_status = True
            print("[HeadsetConnector] Connected to UltraCortex (BrainFlow).")
            return True
        except BrainFlowError as e:
            print(f"[HeadsetConnector] ERROR preparing session: {e}")
            self.connection_status = False
            return False

    def start_stream(self, buffer_size=45000):
        try:
            self.board.start_stream(buffer_size)
            self.streaming = True
            print("[HeadsetConnector] Stream started.")
        except BrainFlowError as e:
            print(f"[HeadsetConnector] ERROR starting stream: {e}")

    def get_data(self, num_samples=250):
        if not self.streaming:
            print("[HeadsetConnector] WARNING: get_data() called but stream not running.")
            return None
        try:
            data = self.board.get_board_data(num_samples)
            if data is None or data.size == 0:
                data = self.board.get_current_board_data(num_samples)
            if data is None or data.size == 0:
                return None
            return data
        except Exception as e:
            print(f"[HeadsetConnector] ERROR reading board data: {e}")
            return None

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

def save_data(self, num_samples=250, filename="data.csv"):
    data = self.get_data(num_samples)
    if data is None:
        print("[HeadsetConnector] No data to save.")
        return

    data = data.T  # rows = samples, columns = channels

    try:
        with open(filename, "w") as f:
            for row in data:
                line = "\t".join(str(val) for val in row)  # tab-separated
                f.write(line + "\n")
        print(f"[HeadsetConnector] Data saved to {filename}.")
    except Exception as e:
        print(f"[HeadsetConnector] ERROR saving data: {e}")
