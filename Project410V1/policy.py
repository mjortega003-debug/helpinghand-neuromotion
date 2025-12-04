from enum import Enum
import numpy as np

class Action(Enum):
    NONE = 0
    LEFT = 1
    RIGHT = 2
    BOTH = 3

def policy_from_intensities(
    intensities: np.ndarray,
    thr: float = 0.3,
    dominance_margin: float = 0.15,
) -> Action:
    """
    intensities: [left, right] in [0,1]
    thr: treshhold, its the minimum intensity to  be consider active
    dominance_margin: how much one side must exceed the other to be 'dominant'
    """
    left, right = float(intensities[0]), float(intensities[1])

    if left < thr and right < thr:
        return Action.NONE

    if left >= thr and right >= thr:
        # both active: check dominance
        if left > right + dominance_margin:
            return Action.LEFT
        elif right > left + dominance_margin:
            return Action.RIGHT
        else:
            return Action.BOTH

    # only one above threshold
    if left >= thr:
        return Action.LEFT
    else:
        return Action.RIGHT
