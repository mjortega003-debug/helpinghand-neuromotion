import datetime
import os

class SessionRecorder:
# Logs intents and commands.
# Later, can export to CSV or SQLite for training.

    def __init__(self, log_path="logs/session.log"):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_path = log_path

    def log(self, intent, command):
        timestamp = datetime.datetime.now().isoformat()
        with open(self.log_path, "a") as f:
            f.write(f"{timestamp}, {intent}, {command}\n")
