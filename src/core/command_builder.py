class CommandBuilder:
# Converts classified intent into a command string.
# Later, can expand to send to Unity or hardware.
    def generate_command(self, intent):
        command_map = {
            "move_left": "CMD_LEFT",
            "move_right": "CMD_RIGHT",
            "grasp": "CMD_GRASP",
            "release": "CMD_RELEASE",
            "idle": "CMD_IDLE"
        }
        return command_map.get(intent, "CMD_UNKNOWN")
