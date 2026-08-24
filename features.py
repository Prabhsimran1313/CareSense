"""
Turns a raw event stream into one feature vector per day, and labels pet events.

Everything downstream reads these daily vectors, never the raw events. Keeping
that boundary clean is what lets the same models run on CASAS data and on your
own LoRaWAN nodes without changes.

Pet handling happens in two places:

  label_pet_events()     heuristic, works on CASAS, needs activity annotations
  resolve_night_motion() beacon based, works on your hardware, no labels needed

Both only ever REMOVE motion from the activity features. Neither is allowed to
add anything to the safety path. Tier A events remain the only thing that can
prove a human is alive and moving, which is why a dog wandering a house all day
still produces an alert.
"""

import numpy as np
import pandas as pd

# Features the rest of the pipeline expects. Order matters for the model input.
DAILY_FEATURE_NAMES = [
    "night_bathroom_trips",
    "first_rise_hour",
    "kitchen_visits",
    "median_transit_seconds",
    "hours_active",
    "longest_inactive_minutes",
    "door_openings",
    "sleep_fragmentation",
    "tier_a_event_count",
]

# Zones where night motion may be attributed to the dog.
#
# Bathroom is deliberately absent. Night bathroom trips are the single strongest
# deterioration signal in the whole system, and if the dog follows the resident
# to the toilet a naive rule would delete exactly the number we care about most.
# Bedroom is absent because motion there during sleep is expected either way.
NIGHT_DOG_ZONES = {"kitchen", "living", "hall", "exit", "unknown"}


def label_pet_events(events):
    """
    Mark motion events that were almost certainly caused by a pet.

    The trick: while the resident is annotated as sleeping, any motion in a room
    other than the bedroom cannot be them. In a single resident home that leaves
    the pet. This gives free training labels without anyone annotating anything.

    Returns the events table with an extra is_pet_candidate column.
    """
    events = events.copy()
    events["is_pet_candidate"] = False

    sleeping_mask = events["activity"].fillna("") == "sleeping"
    away_mask = events["activity"].fillna("") == "leave_home"

    suspicious = (
        (events["sensor_type"] == "motion")
        & (events["zone"] != "bedroom")
        & (sleeping_mask | away_mask)
    )
    events.loc[suspicious, "is_pet_candidate"] = True
    return events


def _nearest_event_gap(query_times, reference_times):
    """Seconds from each query time to the closest reference time."""
    query_times = np.asarray(query_times)
    if len(reference_times) == 0 or len(query_times) == 0:
        return np.full(len(query_times), np.inf)

    reference_times = np.sort(np.asarray(reference_times))
    positions = np.searchsorted(reference_times, query_times)
    positions = np.clip(positions, 1, len(reference_times) - 1)

    gap_before = (query_times - reference_times[positions - 1]) / np.timedelta64(1, "s")
    gap_after = (reference_times[positions] - query_times) / np.timedelta64(1, "s")
    return np.minimum(np.abs(gap_before), np.abs(gap_after))


def _last_bed_state(events, query_times):
    """
    The most recent bed sensor reading before each query time.

    Returns "OCCUPIED", "EMPTY" or "UNKNOWN" per query. CASAS has no bed sensor,
    so this is UNKNOWN everywhere until you run it on your own nodes.
    """
    bed = events[events["sensor_type"] == "bed"].sort_values("timestamp")
    if len(bed) == 0:
        return np.array(["UNKNOWN"] * len(query_times))

    bed_times = bed["timestamp"].values
    bed_states = bed["message"].str.upper().values

    positions = np.searchsorted(bed_times, np.asarray(query_times), side="right") - 1
    return np.where(positions >= 0, bed_states[np.clip(positions, 0, None)], "UNKNOWN")


def resolve_night_motion(
    events,
    night_start_hour=22,
    night_end_hour=6,
    beacon_window_seconds=120,
    tier_a_window_seconds=120,
):
    """
    Use the dog's BLE beacon to attribute night motion.

    The rule: motion in a living area during the small hours, in a room where the
    beacon was also seen, in a single occupancy home, is the dog. At 3am the
    resident is normally in bed, so this inference holds. It does NOT hold during
    the day, when dogs simply follow people around, which is why this only runs
    over the night window.

    No extra sensor is needed. The same nRF52840 that reads the PIR has BLE 5.0
    sitting next to the LoRa radio. Scan for two seconds when the PIR fires, not
    continuously, or the battery life argument disappears.

    Three guards, any of which forces "human":

      1. Zone is bathroom or bedroom  - never attributed to the dog
      2. Bed sensor reads EMPTY       - the resident is demonstrably up
      3. A Tier A event is nearby     - a dog cannot open a door or a fridge

    Adds an is_dog_confirmed column. Returns the events unchanged when there is
    no beacon data, so this is safe to call on CASAS.
    """
    events = events.copy()
    if "is_dog_confirmed" not in events.columns:
        events["is_dog_confirmed"] = False

    beacon = events[events["sensor_type"] == "beacon"]
    if len(beacon) == 0:
        return events

    hours = events["timestamp"].dt.hour
    is_night = (hours >= night_start_hour) | (hours < night_end_hour)
    in_scope = (events["sensor_type"] == "motion") & is_night & events["zone"].isin(NIGHT_DOG_ZONES)

    candidate_index = events.index[in_scope]
    if len(candidate_index) == 0:
        return events

    candidates = events.loc[candidate_index]
    candidate_times = candidates["timestamp"].values

    # Guard 3: a Tier A event close in time means a human was involved.
    tier_a_times = events.loc[events["tier"] == "A", "timestamp"].values
    human_by_tier_a = _nearest_event_gap(candidate_times, tier_a_times) < tier_a_window_seconds

    # Guard 2: bed empty means the resident is up, so night motion is theirs.
    human_by_bed = _last_bed_state(events, candidate_times) == "EMPTY"

    # The beacon test itself, matched within the same zone.
    beacon_nearby = np.zeros(len(candidates), dtype=bool)
    for zone_name in candidates["zone"].unique():
        zone_positions = np.where(candidates["zone"].values == zone_name)[0]
        zone_beacon_times = beacon.loc[beacon["zone"] == zone_name, "timestamp"].values
        gaps = _nearest_event_gap(candidate_times[zone_positions], zone_beacon_times)
        beacon_nearby[zone_positions] = gaps < beacon_window_seconds

    is_dog = beacon_nearby & ~human_by_tier_a & ~human_by_bed
    events.loc[candidate_index[is_dog], "is_dog_confirmed"] = True
    return events


def build_pet_classifier_frame(events):
    """
    Build a training table for a person versus pet motion classifier.

    Positive class is confirmed or candidate pet motion. Negative class is motion
    within two minutes of a Tier A event, which a pet cannot fake.

    Once beacon data exists the positive labels stop being guesses, so this
    becomes properly supervised. Train it in homes that have a tag, then deploy
    in homes that do not - the model learns what dog motion looks like from the
    signature alone, and the tag becomes training scaffolding rather than a
    permanent requirement.
    """
    motion = events[events["sensor_type"] == "motion"].copy()
    tier_a_times = events.loc[events["tier"] == "A", "timestamp"].values

    if len(tier_a_times) == 0 or len(motion) == 0:
        return pd.DataFrame()

    motion["seconds_to_tier_a"] = _nearest_event_gap(motion["timestamp"].values, tier_a_times)
    motion["hour_of_day"] = motion["timestamp"].dt.hour
    motion["seconds_since_previous"] = motion["timestamp"].diff().dt.total_seconds().fillna(999.0)
    motion["zone_changed"] = (motion["zone"] != motion["zone"].shift()).astype(int)

    if "is_pet_candidate" in motion.columns:
        heuristic_pet = motion["is_pet_candidate"].fillna(False)
    else:
        heuristic_pet = pd.Series(False, index=motion.index)

    if "is_dog_confirmed" in motion.columns:
        confirmed_pet = motion["is_dog_confirmed"].fillna(False)
    else:
        confirmed_pet = pd.Series(False, index=motion.index)

    any_pet = heuristic_pet | confirmed_pet
    keep = (motion["seconds_to_tier_a"] < 120) | any_pet

    training_rows = motion[keep].copy()
    training_rows["label_is_pet"] = any_pet[keep].astype(int)
    # Beacon confirmed rows are ground truth. Heuristic rows are inferred.
    training_rows["label_is_certain"] = confirmed_pet[keep].astype(int)
    return training_rows


def _night_window(day_events):
    """Events between midnight and 6am, where night trips are counted."""
    hours = day_events["timestamp"].dt.hour
    return day_events[hours < 6]


def extract_daily_features(events, drop_pet_motion=True, use_beacon=True):
    """
    Collapse the event stream into one row per day.

    drop_pet_motion  False reproduces the naive baseline that treats every motion
                     event as human. Comparing the two is the core experiment.
    use_beacon       False ignores beacon attribution even when tag data exists,
                     so you can measure what the tag actually buys you.

    Note what is NOT filtered: tier_a_event_count comes from the unfiltered
    events on purpose. Pet attribution must never change the safety signal.
    """
    working = events.copy()

    if drop_pet_motion:
        drop_mask = pd.Series(False, index=working.index)
        if "is_pet_candidate" in working.columns:
            drop_mask = drop_mask | working["is_pet_candidate"].fillna(False)
        if use_beacon and "is_dog_confirmed" in working.columns:
            drop_mask = drop_mask | working["is_dog_confirmed"].fillna(False)
        working = working[~drop_mask]

    # Tier A counts are taken from the untouched table, not the filtered one.
    tier_a_per_day = events[events["tier"] == "A"].groupby("date").size()

    daily_rows = []
    for day, day_events in working.groupby("date"):
        day_events = day_events.sort_values("timestamp")
        binary_events = day_events[~day_events["sensor_type"].isin(["temperature", "beacon"])]

        if len(binary_events) < 5:
            continue

        night_events = _night_window(binary_events)
        bathroom_flags = (night_events["zone"] == "bathroom").astype(int)
        night_bathroom_trips = int(bathroom_flags.diff().fillna(bathroom_flags).eq(1).sum())

        # First activity outside the bedroom after 4am counts as getting up.
        morning = binary_events[
            (binary_events["timestamp"].dt.hour >= 4) & (binary_events["zone"] != "bedroom")
        ]
        if len(morning) == 0:
            first_rise_hour = np.nan
        else:
            rise_time = morning["timestamp"].iloc[0]
            first_rise_hour = rise_time.hour + rise_time.minute / 60.0

        kitchen_events = binary_events[binary_events["zone"] == "kitchen"]
        kitchen_gaps = kitchen_events["timestamp"].diff().dt.total_seconds()
        kitchen_visits = int((kitchen_gaps > 1800).sum() + (1 if len(kitchen_events) else 0))

        # Transit time is the gap between events in two different zones.
        zone_changes = binary_events[binary_events["zone"] != binary_events["zone"].shift()]
        transit_gaps = zone_changes["timestamp"].diff().dt.total_seconds()
        transit_gaps = transit_gaps[(transit_gaps > 1) & (transit_gaps < 300)]
        median_transit_seconds = float(transit_gaps.median()) if len(transit_gaps) else np.nan

        all_gaps = binary_events["timestamp"].diff().dt.total_seconds().fillna(0)
        longest_inactive_minutes = float(all_gaps.max() / 60.0)

        active_span = binary_events["timestamp"].iloc[-1] - binary_events["timestamp"].iloc[0]
        hours_active = active_span.total_seconds() / 3600.0

        door_openings = int((day_events["sensor_type"] == "door").sum())

        bedroom_exits = night_events[night_events["zone"] != "bedroom"]
        sleep_fragmentation = int(
            (bedroom_exits["timestamp"].diff().dt.total_seconds().fillna(9999) > 600).sum()
        )

        daily_rows.append(
            {
                "date": day,
                "night_bathroom_trips": night_bathroom_trips,
                "first_rise_hour": first_rise_hour,
                "kitchen_visits": kitchen_visits,
                "median_transit_seconds": median_transit_seconds,
                "hours_active": hours_active,
                "longest_inactive_minutes": longest_inactive_minutes,
                "door_openings": door_openings,
                "sleep_fragmentation": sleep_fragmentation,
                "tier_a_event_count": int(tier_a_per_day.get(day, 0)),
            }
        )

    daily = pd.DataFrame(daily_rows).sort_values("date").reset_index(drop=True)
    # Forward fill the odd missing value rather than dropping a whole day.
    for column in DAILY_FEATURE_NAMES:
        daily[column] = daily[column].ffill().bfill()
    return daily