import csv
import os
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np


@dataclass
class DataSaverConfig:
    # Folder for all outputs
    logs_dir: str = "logs"

    # Output filenames (suffix added if unique_per_run=True)
    raw_csv_name: str = "neuromotion_raw_eeg.csv"
    processed_csv_name: str = "neuromotion_processed.csv"

    # Toggle outputs
    save_raw: bool = True
    save_processed: bool = True

    # If True, create new files each run
    unique_per_run: bool = True

    # Metadata
    input_source: str = "unknown"  # mock, ultracortex, lsl, etc.
    sample_rate: Optional[int] = None
    channel_count: Optional[int] = None


class DataSaver:
    """
    Saver script:
    - Creates log files in logs/
    - Writes raw EEG rows continuously (timestamp + ch_0..ch_{N-1})
    - Writes processed rows (timestamp + intent + command + feature vector)

    This module should only save. It should not filter or classify.
    """

    def __init__(self, config: Optional[DataSaverConfig] = None):
        self.cfg = config or DataSaverConfig()
        os.makedirs(self.cfg.logs_dir, exist_ok=True)

        suffix = ""
        if self.cfg.unique_per_run:
            suffix = time.strftime("_%Y%m%d_%H%M%S")

        self.raw_path = os.path.join(
            self.cfg.logs_dir,
            self.cfg.raw_csv_name.replace(".csv", f"{suffix}.csv")
        )
        self.processed_path = os.path.join(
            self.cfg.logs_dir,
            self.cfg.processed_csv_name.replace(".csv", f"{suffix}.csv")
        )

        self._raw_file = None
        self._raw_writer = None
        self._raw_header_written = False

        self._proc_file = None
        self._proc_writer = None
        self._proc_header_written = False

    def start(self) -> None:
        """Open files and prepare writers."""
        if self.cfg.save_raw:
            self._raw_file = open(self.raw_path, "w", newline="")
            self._raw_writer = csv.writer(self._raw_file)

        if self.cfg.save_processed:
            self._proc_file = open(self.processed_path, "w", newline="")
            self._proc_writer = csv.writer(self._proc_file)

        print(f"[DataSaver] Raw output: {self.raw_path if self.cfg.save_raw else 'off'}")
        print(f"[DataSaver] Processed output: {self.processed_path if self.cfg.save_processed else 'off'}")

    def close(self) -> None:
        """Flush and close files."""
        try:
            if self._raw_file:
                self._raw_file.flush()
                self._raw_file.close()
        finally:
            self._raw_file = None
            self._raw_writer = None

        try:
            if self._proc_file:
                self._proc_file.flush()
                self._proc_file.close()
        finally:
            self._proc_file = None
            self._proc_writer = None

        print("[DataSaver] Closed.")

    def save_raw_chunk(
        self,
        eeg: Union[np.ndarray, Sequence[Sequence[float]]],
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Save raw EEG in a wide CSV format.

        Expected shapes:
        - (channels, samples) preferred
        - (samples, channels) also accepted (will transpose)

        Each CSV row:
        timestamp, input_source, ch_0, ch_1, ... ch_{N-1}
        """
        if not self.cfg.save_raw or self._raw_writer is None:
            return

        ts = time.time() if timestamp is None else timestamp
        arr = np.asarray(eeg)

        if arr.ndim != 2 or arr.size == 0:
            return

        # If likely (samples, channels), transpose to (channels, samples)
        if arr.shape[0] > arr.shape[1] and arr.shape[1] <= 32:
            arr = arr.T

        channels, samples = arr.shape
        self.cfg.channel_count = self.cfg.channel_count or channels

        if not self._raw_header_written:
            header = ["timestamp", "input_source"] + [f"ch_{i}" for i in range(channels)]
            self._raw_writer.writerow(header)
            self._raw_header_written = True

        if self.cfg.sample_rate and self.cfg.sample_rate > 0:
            dt = 1.0 / float(self.cfg.sample_rate)
            start_ts = ts - (samples - 1) * dt
            for j in range(samples):
                row_ts = start_ts + j * dt
                row = [row_ts, self.cfg.input_source] + [float(arr[i, j]) for i in range(channels)]
                self._raw_writer.writerow(row)
        else:
            for j in range(samples):
                row = [ts, self.cfg.input_source] + [float(arr[i, j]) for i in range(channels)]
                self._raw_writer.writerow(row)

    def save_processed(
        self,
        intent: str,
        command: str,
        features: Optional[Union[np.ndarray, Sequence[float]]] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Save processed outputs:
        timestamp, input_source, intent, command, feat_0..feat_k (optional)
        """
        if not self.cfg.save_processed or self._proc_writer is None:
            return

        ts = time.time() if timestamp is None else timestamp

        feat_list = None
        if features is not None:
            feat_arr = np.asarray(features).reshape(-1)
            feat_list = [float(x) for x in feat_arr.tolist()]

        if not self._proc_header_written:
            header = ["timestamp", "input_source", "intent", "command"]
            if feat_list is not None:
                header += [f"feat_{i}" for i in range(len(feat_list))]
            self._proc_writer.writerow(header)
            self._proc_header_written = True

        row = [ts, self.cfg.input_source, intent, command]
        if feat_list is not None:
            row += feat_list

        self._proc_writer.writerow(row)