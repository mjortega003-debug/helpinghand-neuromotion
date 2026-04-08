from stable_baselines3 import PPO
import pandas as pd
import os

class IntentDetector:
    def __init__(self, model_path=r"src\models\ppo_gesture_hand.zip", csv_path=r"logs\neuromotion_data.csv"):
        self.model_path = model_path
        self.csv_path = csv_path
        self.model = None
        self.labels = []
        
        self.load_labels()
        self.load_model()

    def load_labels(self):
        """Peeks at the training data to ensure actions match the correct gesture names."""
        if os.path.exists(self.csv_path):
            try:
                df = pd.read_csv(self.csv_path)
                
                df = df[df['cv_label'] != 'none']
   
                df = df.dropna(subset=['cv_label'])

                self.labels = [str(label) for label in df['cv_label'].unique().tolist()]
                
                print(f"[IntentDetector] Mapped AI Actions to Labels: {self.labels}")
            except Exception as e:
                print(f"[IntentDetector] Error loading labels from CSV: {e}")
        else:
            print("[IntentDetector] WARNING: CSV not found. Cannot map labels.")

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = PPO.load(self.model_path, device="cpu")
                print(f"[IntentDetector] Brainwave model loaded: {self.model_path}")
            except Exception as e:
                print(f"[IntentDetector] Model load failed: {e}")
        else:
            print("[IntentDetector] WARNING: No model found. Run train_ppo.py first.")

    def classify(self, rms_features):

        if self.model and self.labels:
            action, _states = self.model.predict(rms_features, deterministic=True)
            
            try:
                return str(self.labels[int(action)])
            except Exception:
                return "unknown"
        
        return "idle"