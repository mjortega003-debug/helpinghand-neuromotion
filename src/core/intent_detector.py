import random

class IntentDetector:
# Mock classifier. Later, replace with trained ML model.
    def classify(self, features):
    # For now, returns a random intent.
        intents = ["move_left", "move_right", "grasp", "release", "idle"]
        return random.choice(intents)
