"""
Runs forever in the background, writing fake sensor events to sensor_log.csv,
one at a time, like a real Pi relaying LoRaWAN packets would.

This process knows nothing about ML, features, or alerts. It only produces
raw events. That separation matters: it's exactly the boundary that will
exist between the Pi guy's code and yours, so building the demo this way
tests the real architecture, not a shortcut.

Run it in its own terminal tab and leave it running:

    python simulator.py --mode normal
    python simulator.py --mode emergency
    python simulator.py --mode decline
"""

import argparse
import csv
import random
import time
from datetime import datetime
from pathlib import Path

LOG_PATH = Path("sensor_log.csv")

SENSORS = {
    "bedroom-motion": "motion",
    "bathroom-motion": "motion",
    "kitchen-motion": "motion",
    "living-motion": "motion",
    "front-door": "door",
    "fridge-door": "door",
    "bed-mat": "bed",
    "living-temp": "temperature",
}


def ensure_log_exists():
    if not LOG_PATH.exists():
        with open(LOG_PATH, "w", newline="") as handle:
            csv.writer(handle).writerow(["timestamp", "sensor_id", "value"])


def emit(sensor_id, value):
    with open(LOG_PATH, "a", newline="") as handle:
        csv.writer(handle).writerow(
            [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), sensor_id, value]
        )
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {sensor_id}: {value}")


def random_reading(sensor_type):
    if sensor_type == "motion":
        return "ON"
    if sensor_type == "door":
        return random.choice(["OPEN", "CLOSE"])
    if sensor_type == "bed":
        return random.choice(["OCCUPIED", "EMPTY"])
    return round(20 + random.uniform(-1.5, 1.5), 1)


def run_normal(interval_seconds):
    """Ordinary background activity. This is the 'nothing's wrong' state."""
    print("simulating NORMAL activity — press Ctrl+C to stop")
    while True:
        sensor_id = random.choice(list(SENSORS))
        emit(sensor_id, random_reading(SENSORS[sensor_id]))
        time.sleep(interval_seconds)


def run_decline(interval_seconds):
    """
    Activity that's technically present but visibly thinner: fewer kitchen
    trips, fewer door events, longer gaps. This is what the trend detector
    should catch over several minutes of accumulated log, not the watchdog.
    """
    print("simulating a QUIET/DECLINING day — press Ctrl+C to stop")
    while True:
        sensor_id = random.choices(
            list(SENSORS),
            weights=[3 if SENSORS[s] == "motion" else 1 for s in SENSORS],
        )[0]
        emit(sensor_id, random_reading(SENSORS[sensor_id]))
        time.sleep(interval_seconds * 2.5)  # everything happens more slowly


def run_emergency(interval_seconds):
    """
    The scenario to trigger live on stage. Motion keeps firing (a pet, or
    just noise) but no Tier A event appears at all — no door, no fridge,
    no bed state change. That absence is the actual alert condition.
    """
    print("simulating an EMERGENCY — motion only, no confirmed human events")
    while True:
        emit(random.choice(["kitchen-motion", "living-motion"]), "ON")
        time.sleep(interval_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["normal", "decline", "emergency"], default="normal")
    parser.add_argument("--interval", type=float, default=3.0, help="seconds between events")
    arguments = parser.parse_args()

    ensure_log_exists()

    if arguments.mode == "normal":
        run_normal(arguments.interval)
    elif arguments.mode == "decline":
        run_decline(arguments.interval)
    else:
        run_emergency(arguments.interval)
