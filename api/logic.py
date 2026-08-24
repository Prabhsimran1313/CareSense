"""
All the actual logic, kept separate from FastAPI on purpose.

Testing this file needs nothing but pandas. Testing api.py needs a running
server. Keeping them apart means you can trust this file works before you
ever start uvicorn, which matters when you're debugging under time pressure.
"""

import re
from datetime import datetime, timedelta
from pathlib import Path

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
SENSOR_ROOM = {
    "bedroom-motion": "bedroom",
    "bathroom-motion": "bathroom",
    "kitchen-motion": "kitchen",
    "living-motion": "living",
    "front-door": "hall",
    "fridge-door": "kitchen",
    "bed-mat": "bedroom",
    "living-temp": "living",
}


def load_log():
    if not LOG_PATH.exists():
        return pd.DataFrame(columns=["timestamp", "sensor_id", "value"])
    log = pd.read_csv(LOG_PATH, parse_dates=["timestamp"])
    log["sensor_type"] = log["sensor_id"].map(SENSOR_TYPE).fillna("other")
    log["room"] = log["sensor_id"].map(SENSOR_ROOM).fillna("unknown")
    return log.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Chart data — each function returns plain dicts/lists, ready for JSON.
# ---------------------------------------------------------------------------

def get_current_status(log, silence_limit_seconds=20, minimum_tier_a=1, window_minutes=10):
    """Same rule as the demo watchdog, returned as structured data for a UI."""
    if len(log) == 0:
        return {"alert_level": "silent", "message": "No data yet.", "last_event": None}

    cutoff = datetime.now() - timedelta(minutes=window_minutes)
    recent = log[log["timestamp"] >= cutoff]
    last_event = log.iloc[-1]

    if len(recent) == 0:
        minutes_ago = (datetime.now() - last_event["timestamp"]).total_seconds() / 60.0
        return {
            "alert_level": "urgent",
            "message": f"No events in the last {window_minutes} minutes.",
            "last_event": {"sensor_id": last_event["sensor_id"], "minutes_ago": round(minutes_ago, 1)},
        }

    tier_a_count = int(recent["sensor_type"].isin(TIER_A_TYPES).sum())
    motion_count = int((recent["sensor_type"] == "motion").sum())
    window_span = (recent["timestamp"].max() - recent["timestamp"].min()).total_seconds()

    sustained_zero_tier_a = tier_a_count < minimum_tier_a and window_span >= silence_limit_seconds
    if sustained_zero_tier_a:
        level = "urgent"
        message = f"{motion_count} motion events but 0 confirmed human events across {window_span:.0f}s."
    elif tier_a_count == 0:
        level = "nudge"
        message = f"{motion_count} motion events, no confirmed human activity yet."
    else:
        level = "silent"
        message = f"{tier_a_count} confirmed human events in the last {window_minutes} minutes. Normal."

    return {
        "alert_level": level,
        "message": message,
        "last_event": {
            "sensor_id": last_event["sensor_id"],
            "room": last_event["room"],
            "seconds_ago": round((datetime.now() - last_event["timestamp"]).total_seconds(), 0),
        },
    }


def get_room_activity(log):
    """Last-seen time per room, for the room grid on the status screen."""
    if len(log) == 0:
        return []
    rooms = []
    for room, group in log.groupby("room"):
        last = group.iloc[-1]
        seconds_ago = (datetime.now() - last["timestamp"]).total_seconds()
        rooms.append({"room": room, "seconds_ago": round(seconds_ago, 0), "sensor_id": last["sensor_id"]})
    return sorted(rooms, key=lambda r: r["seconds_ago"])


def get_activity_trend(log, bucket_minutes=1, window_minutes=30):
    """
    Event counts bucketed over time, ready for a sparkline or bar chart.

    In the real system this would be daily buckets over 28 days. For a live
    demo the clock is compressed to minutes so a chart actually moves during
    a five minute pitch.
    """
    if len(log) == 0:
        return []
    cutoff = datetime.now() - timedelta(minutes=window_minutes)
    recent = log[log["timestamp"] >= cutoff].copy()
    if len(recent) == 0:
        return []

    recent["bucket"] = recent["timestamp"].dt.floor(f"{bucket_minutes}min")
    counts = recent.groupby(["bucket", "sensor_type"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=["motion", "door", "bed", "temperature"], fill_value=0)

    return [
        {"time": str(bucket), **row.to_dict()}
        for bucket, row in counts.iterrows()
    ]


def get_naive_vs_personalised(log, window_minutes=30):
    """
    The comparison chart: how often would a motion-only system have alerted,
    versus a system that requires confirmed Tier A activity.
    """
    if len(log) == 0:
        return {"naive_alerts": 0, "personalised_alerts": 0, "buckets_checked": 0}

    cutoff = datetime.now() - timedelta(minutes=window_minutes)
    recent = log[log["timestamp"] >= cutoff].copy()
    if len(recent) == 0:
        return {"naive_alerts": 0, "personalised_alerts": 0, "buckets_checked": 0}

    recent["bucket"] = recent["timestamp"].dt.floor("1min")
    naive_alerts, personalised_alerts = 0, 0
    for _, bucket_events in recent.groupby("bucket"):
        has_motion = (bucket_events["sensor_type"] == "motion").any()
        has_tier_a = bucket_events["sensor_type"].isin(TIER_A_TYPES).any()
        if not has_motion:
            naive_alerts += 1  # naive system alerts on any quiet minute
        if not has_tier_a:
            personalised_alerts += 1  # personalised system needs confirmed activity

    return {
        "naive_alerts": naive_alerts,
        "personalised_alerts": personalised_alerts,
        "buckets_checked": recent["bucket"].nunique(),
    }


# ---------------------------------------------------------------------------
# Gap detection — the "stuck in one room too long" warning
# ---------------------------------------------------------------------------

def find_room_gaps(log, room_limits_minutes=None):
    """
    Flags a room where the most recent motion is older than its limit, given
    that some other room has been active since — meaning the person moved on
    but a sensor never confirmed them leaving, OR the same room has shown no
    exit signal for longer than expected.

    Default limits are demo-scale minutes. In the real system these would be
    hours (e.g. bedroom > 10 hours with no other room activity is the actual
    example you gave — same logic, compressed clock for a live demo).
    """
    if room_limits_minutes is None:
        room_limits_minutes = {
            "bedroom": 8,      # compressed stand-in for "10 hours" in the real system
            "bathroom": 3,
            "kitchen": 5,
            "living": 6,
        }

    if len(log) == 0:
        return []

    warnings = []
    now = datetime.now()
    most_recent_room = log.iloc[-1]["room"]

    for room, limit_minutes in room_limits_minutes.items():
        room_events = log[log["room"] == room]
        if len(room_events) == 0:
            continue

        last_seen = room_events.iloc[-1]["timestamp"]
        minutes_since = (now - last_seen).total_seconds() / 60.0

        # Only meaningful if this room is where the person currently is —
        # i.e. it's the most recently active room overall.
        if room == most_recent_room:
            other_room_events = log[log["room"] != room]
            if len(other_room_events) > 0:
                last_other = other_room_events.iloc[-1]["timestamp"]
                stay_start = max(last_other, room_events.iloc[0]["timestamp"])
            else:
                stay_start = room_events.iloc[0]["timestamp"]
            stay_minutes = (now - stay_start).total_seconds() / 60.0

            if stay_minutes >= limit_minutes:
                warnings.append(
                    {
                        "room": room,
                        "minutes_in_room": round(stay_minutes, 1),
                        "limit_minutes": limit_minutes,
                        "message": f"No activity outside {room} for {stay_minutes:.0f} minutes "
                        f"(expected within {limit_minutes}).",
                    }
                )

    return warnings


# ---------------------------------------------------------------------------
# LLM Q&A — question-aware chunking, never the whole file
# ---------------------------------------------------------------------------

TIME_PATTERNS = [
    (re.compile(r"last (\d+) ?min"), lambda m: int(m.group(1))),
    (re.compile(r"last (\d+) ?hour"), lambda m: int(m.group(1)) * 60),
    (re.compile(r"today"), lambda m: 24 * 60),
    (re.compile(r"hour"), lambda m: 60),
]

ROOM_KEYWORDS = ["bedroom", "bathroom", "kitchen", "living", "hall"]
SENSOR_TYPE_KEYWORDS = {"motion": "motion", "door": "door", "temperature": "temperature", "bed": "bed"}


def extract_relevant_chunk(log, question, default_window_minutes=15, max_rows=60):
    """
    Turns a natural language question into a small, targeted slice of the
    log — never the whole file. Two filters, both optional and combinable:

      time window   "in the last 30 minutes", "today" -> narrows by time
      room / sensor "in the kitchen", "the door"       -> narrows by room/type

    If nothing matches, falls back to a short recent window. Returns the
    filtered rows plus a plain-text description of what was selected, so the
    LLM prompt can say what it's looking at instead of guessing.
    """
    question_lower = question.lower()

    window_minutes = default_window_minutes
    for pattern, extractor in TIME_PATTERNS:
        match = pattern.search(question_lower)
        if match:
            window_minutes = extractor(match)
            break

    cutoff = datetime.now() - timedelta(minutes=window_minutes)
    chunk = log[log["timestamp"] >= cutoff]

    matched_room = next((r for r in ROOM_KEYWORDS if r in question_lower), None)
    if matched_room:
        chunk = chunk[chunk["room"] == matched_room]

    matched_type = next((t for kw, t in SENSOR_TYPE_KEYWORDS.items() if kw in question_lower), None)
    if matched_type:
        chunk = chunk[chunk["sensor_type"] == matched_type]

    # Cap it. An LLM prompt does not need thousands of rows to answer a
    # question about the last half hour — and a hard cap keeps token use
    # and latency predictable during a demo.
    chunk = chunk.tail(max_rows)

    description_parts = [f"last {window_minutes} minute(s)"]
    if matched_room:
        description_parts.append(f"room: {matched_room}")
    if matched_type:
        description_parts.append(f"sensor type: {matched_type}")

    return chunk, ", ".join(description_parts)


def chunk_to_text(chunk):
    """Renders a filtered chunk as compact text for the LLM prompt."""
    if len(chunk) == 0:
        return "(no matching events)"
    lines = [
        f"{row.timestamp.strftime('%H:%M:%S')} {row.room} {row.sensor_id} {row.value}"
        for row in chunk.itertuples()
    ]
    return "\n".join(lines)
