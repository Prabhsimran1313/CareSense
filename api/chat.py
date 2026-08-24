"""
The 'chatbot' side of the demo.

Runs separately from simulator.py, in its own terminal tab. It never writes
to the log, only reads it — same as the real model will only ever read what
the Pi produces. Every time you ask it something, it re-reads the CURRENT
state of sensor_log.csv, so it always reflects whatever the simulator has
written so far. That live re-read is what makes this feel like a running
system rather than a one-off script.

Two commands:
    status   - summarise activity in the log so far, statistically
    check    - run it through the real detector logic and get an alert

Usage:
    python chat.py
    > status
    > check
    > quit
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

LOG_PATH = Path("sensor_log.csv")

TIER_A_TYPES = {"door", "bed"}
SENSOR_TYPE = {
    "bedroom-motion": "motion",
    "bathroom-motion": "motion",
    "kitchen-motion": "motion",
    "living-motion": "motion",
    "front-door": "door",
    "fridge-door": "door",
    "bed-mat": "bed",
    "living-temp": "temperature",
}


def load_log():
    if not LOG_PATH.exists():
        return pd.DataFrame(columns=["timestamp", "sensor_id", "value"])
    log = pd.read_csv(LOG_PATH, parse_dates=["timestamp"])
    log["sensor_type"] = log["sensor_id"].map(SENSOR_TYPE).fillna("other")
    return log


def summarise_statistically(log, window_minutes=10):
    """
    Plain statistics over the most recent window — no ML, just counts and
    gaps. This is what a 'status' question answers: what has actually been
    seen so far, described in numbers.
    """
    if len(log) == 0:
        return "No events logged yet — is the simulator running?"

    cutoff = datetime.now() - timedelta(minutes=window_minutes)
    recent = log[log["timestamp"] >= cutoff]

    if len(recent) == 0:
        last_seen = log["timestamp"].max()
        minutes_ago = (datetime.now() - last_seen).total_seconds() / 60.0
        return f"No events in the last {window_minutes} minutes. Last event was {minutes_ago:.1f} minutes ago."

    motion_count = int((recent["sensor_type"] == "motion").sum())
    tier_a_count = int(recent["sensor_type"].isin(TIER_A_TYPES).sum())
    per_sensor = recent["sensor_id"].value_counts().to_dict()

    gaps = recent["timestamp"].sort_values().diff().dt.total_seconds().dropna()
    longest_gap_seconds = float(gaps.max()) if len(gaps) else 0.0

    lines = [
        f"Last {window_minutes} min: {len(recent)} events total.",
        f"  motion events: {motion_count}",
        f"  confirmed human events (door/bed): {tier_a_count}",
        f"  longest gap between events: {longest_gap_seconds:.0f}s",
        f"  by sensor: {per_sensor}",
    ]
    return "\n".join(lines)


def check_for_alert(log, window_minutes=10, silence_limit_seconds=20, minimum_tier_a=1):
    """
    The scaled-down version of AbsenceWatchdog for a live demo timescale.

    The real system uses hours and days. A demo can't wait hours, so the
    same logic runs on minutes instead — same rule, compressed clock:
    motion present but confirmed human events (door, bed) absent or sparse
    across the whole window is worth flagging, regardless of how busy the
    motion sensor has been. A dog can keep the PIR firing every second and
    that must never look like a quiet, safe day.
    """
    if len(log) == 0:
        return "silent", "No data yet."

    cutoff = datetime.now() - timedelta(minutes=window_minutes)
    recent = log[log["timestamp"] >= cutoff]

    if len(recent) == 0:
        return "urgent", f"No events at all in the last {window_minutes} minutes."

    tier_a_count = int(recent["sensor_type"].isin(TIER_A_TYPES).sum())
    motion_count = int((recent["sensor_type"] == "motion").sum())

    window_span_seconds = (recent["timestamp"].max() - recent["timestamp"].min()).total_seconds()
    gaps = recent["timestamp"].sort_values().diff().dt.total_seconds().dropna()
    longest_gap = float(gaps.max()) if len(gaps) else 0.0

    # Two ways to be silent on Tier A: one long gap, or zero Tier A across a
    # window that has been running long enough to expect at least one.
    sustained_zero_tier_a = tier_a_count < minimum_tier_a and window_span_seconds >= silence_limit_seconds

    if sustained_zero_tier_a or longest_gap >= silence_limit_seconds:
        return (
            "urgent",
            f"{motion_count} motion events but only {tier_a_count} confirmed human "
            f"event(s) across {window_span_seconds:.0f}s. Motion alone does not confirm "
            "a person — could be a pet, could be nothing. This would trigger an absence alert.",
        )

    if tier_a_count == 0:
        return "nudge", f"{motion_count} motion events, no confirmed human activity yet. Watching."

    return "silent", f"{tier_a_count} confirmed human events in the last {window_minutes} minutes. Normal."


def print_help():
    print("commands: status | check | help | quit")


def main():
    print("Quiet Care — live demo console")
    print(f"reading from {LOG_PATH.resolve()}")
    print_help()

    while True:
        try:
            command = input("\n> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if command in ("quit", "exit"):
            break
        if command == "help":
            print_help()
            continue

        log = load_log()  # re-read every time, so it reflects the live simulator

        if command == "status":
            print(summarise_statistically(log))
        elif command == "check":
            level, message = check_for_alert(log)
            print(f"[{level.upper()}] {message}")
        else:
            print("unknown command.")
            print_help()


if __name__ == "__main__":
    sys.exit(main())
