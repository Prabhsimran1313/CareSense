"""
Loads raw CASAS smart home data into a tidy event table.

CASAS files look like this, one event per line:
    2010-11-04 00:03:50.209589   M003   ON
    2010-11-04 00:22:59.984654   M003   ON   Sleeping   begin

The last two fields only appear when an activity annotation starts or ends.

This module does two useful things beyond parsing:
  1. It works out which functional zone each sensor belongs to, by looking at
     which activity labels the sensor fires during. That means you do not need
     a floorplan, and the same code works for Aruba, Milan or any other home.
  2. It splits sensors into evidence tiers. Tier A sensors are ones a pet
     cannot trigger, which matters a lot for the safety logic later on.
"""

import pandas as pd

# Activity labels are grouped into zones. Anything not listed stays "unknown".
ACTIVITY_TO_ZONE = {
    "sleeping": "bedroom",
    "bed_to_toilet": "bathroom",
    "bathing": "bathroom",
    "master_bathroom": "bathroom",
    "guest_bathroom": "bathroom",
    "wash_bathtub": "bathroom",
    "morning_meds": "bathroom",
    "eve_meds": "bathroom",
    "take_medicine": "kitchen",
    "meal_preparation": "kitchen",
    "cook": "kitchen",
    "kitchen_activity": "kitchen",
    "eating": "kitchen",
    "eat": "kitchen",
    "wash_dishes": "kitchen",
    "relax": "living",
    "watch_tv": "living",
    "read": "living",
    "desk_activity": "living",
    "work": "living",
    "chores": "living",
    "housekeeping": "living",
    "leave_home": "exit",
    "enter_home": "exit",
    "respirate": "living",
    "meditate": "living",
}


def read_raw_events(file_path):
    """Parse a CASAS data file into a DataFrame of events."""
    parsed_rows = []
    open_activity = None

    with open(file_path, "r", errors="ignore") as handle:
        for raw_line in handle:
            parts = raw_line.split()
            if len(parts) < 4:
                continue

            date_text, time_text, sensor_id, message = parts[0], parts[1], parts[2], parts[3]

            # Track which activity is currently running, so every event gets a label.
            if len(parts) >= 6:
                activity_name = parts[4].strip().lower()
                marker = parts[5].strip().lower()
                if marker.startswith("begin"):
                    open_activity = activity_name
                elif marker.startswith("end"):
                    open_activity = None

            timestamp = pd.to_datetime(f"{date_text} {time_text}", errors="coerce")
            if pd.isna(timestamp):
                continue

            parsed_rows.append(
                {
                    "timestamp": timestamp,
                    "sensor_id": sensor_id,
                    "message": message,
                    "activity": open_activity,
                }
            )

    events = pd.DataFrame(parsed_rows)
    events["sensor_type"] = events["sensor_id"].str[0].map(
        {
            "M": "motion",
            "D": "door",
            "T": "temperature",
            "I": "item",
            "L": "light",
            "A": "device",
            "B": "beacon",  # dog collar tag, read over BLE by the same node
            "P": "bed",     # bed occupancy, OCCUPIED or EMPTY
        }
    )
    events["sensor_type"] = events["sensor_type"].fillna("other")
    return events.sort_values("timestamp").reset_index(drop=True)


def infer_sensor_zones(events, minimum_events=20):
    """
    Work out a room zone for each sensor from the activity labels it fires during.

    This is data driven on purpose. Hardcoding a floorplan means the code only
    works for one house, and you want to train on Aruba and test on Milan.
    """
    labelled = events.dropna(subset=["activity"]).copy()
    labelled["zone"] = labelled["activity"].map(ACTIVITY_TO_ZONE)
    labelled = labelled.dropna(subset=["zone"])

    zone_counts = labelled.groupby(["sensor_id", "zone"]).size().rename("count").reset_index()

    sensor_zone_map = {}
    for sensor_id, group in zone_counts.groupby("sensor_id"):
        if group["count"].sum() < minimum_events:
            continue
        best_zone = group.sort_values("count", ascending=False).iloc[0]["zone"]
        sensor_zone_map[sensor_id] = best_zone

    # Door sensors are almost always entry doors in these homes.
    for sensor_id in events.loc[events["sensor_type"] == "door", "sensor_id"].unique():
        sensor_zone_map.setdefault(sensor_id, "exit")

    # Beacons never co-occur with activity labels, so they cannot be placed by
    # the inference above. Convention: a beacon inherits the zone of the motion
    # sensor with the same number, so B005 sits in the same room as M005.
    for sensor_id in events.loc[events["sensor_type"] == "beacon", "sensor_id"].unique():
        paired_motion = "M" + sensor_id[1:]
        if paired_motion in sensor_zone_map:
            sensor_zone_map[sensor_id] = sensor_zone_map[paired_motion]

    # Bed sensors are in the bedroom by definition.
    for sensor_id in events.loc[events["sensor_type"] == "bed", "sensor_id"].unique():
        sensor_zone_map.setdefault(sensor_id, "bedroom")

    return sensor_zone_map


def assign_evidence_tier(sensor_id, sensor_type, zone):
    """
    Tier A sensors cannot plausibly be triggered by a pet.

    A dog does not open a door, use a toilet, or operate an appliance. Tier A
    events are therefore the only ones the safety watchdog is allowed to trust.
    Tier B still feeds the activity and mobility features, it just does not get
    a vote on whether someone is in trouble.
    """
    if sensor_type in ("door", "item", "device", "light"):
        return "A"
    if sensor_type == "bed":
        return "A"
    if sensor_type == "beacon":
        # Evidence about the dog, never about the human. Must not be Tier A.
        return "pet"
    if sensor_type == "temperature":
        return "context"
    if zone == "bathroom":
        # Bathroom motion is the closest proxy CASAS gives us to a flush sensor.
        return "A"
    return "B"


def load_home(file_path):
    """Load one CASAS home and return the enriched event table plus its zone map."""
    events = read_raw_events(file_path)
    sensor_zone_map = infer_sensor_zones(events)

    events["zone"] = events["sensor_id"].map(sensor_zone_map).fillna("unknown")
    events["tier"] = [
        assign_evidence_tier(sensor_id, sensor_type, zone)
        for sensor_id, sensor_type, zone in zip(events["sensor_id"], events["sensor_type"], events["zone"])
    ]
    events["date"] = events["timestamp"].dt.normalize()
    return events, sensor_zone_map