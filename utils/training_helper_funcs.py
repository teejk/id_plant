import math
import time


def time_elapsed_remaining(initial_time: int, progress: float) -> str:
    """Generate a string with the elapsed time and the estimated remaining time.

    Parameters
    ----------
    initial_time : int
        Time when the training started in seconds (UNIX timestamp)

    percent : float
        Percentage of the training completed

    Returns
    -------
    str
        String with the elapsed time and the estimated remaining time
    """

    def as_minutes(seconds):
        minutes = math.floor(seconds / 60)
        seconds -= minutes * 60
        return f"{minutes}m {seconds}s"

    now = time.time()
    seconds = now - initial_time
    estimated_total = seconds / (progress)
    estimated_remaining = estimated_total - seconds
    return (
        f"Elapsed: {as_minutes(seconds)}, Remaining: {as_minutes(estimated_remaining)}"
    )
