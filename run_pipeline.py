"""
Runs the whole pipeline on a CASAS home.

    python run_pipeline.py --data data/milan/data --home milan
    python run_pipeline.py --data data/aruba/data --home aruba --no-llm

Add --compare-naive to print the experiment that matters: what happens to a
motion-only system when there is a dog in the house.
"""

import argparse
import json
import joblib

import pandas as pd

from casas_loader import load_home
from detectors import (
    AbsenceWatchdog,
    AcuteDetector,
    PersonalBaseline,
    SimpleThresholdDetector,
    TrendAnalyser,
)
from features import (
    build_pet_classifier_frame,
    extract_daily_features,
    label_pet_events,
    resolve_night_motion,
)
from narrative import write_update


def summarise_alert(day_row, absence_row, trend_findings, anomaly_score):
    """Pack detector output into the small dict the narrative layer expects."""
    notable = [
        row
        for row in trend_findings.to_dict("records")
        if abs(row["percent_change_per_week"]) > 5.0
    ][:2]

    if absence_row["absence_alert"]:
        alert_level = "urgent"
    elif anomaly_score > 0:
        alert_level = "nudge"
    elif notable:
        alert_level = "nudge"
    else:
        alert_level = "silent"

    return {
        "date": str(day_row["date"].date()),
        "alert_level": alert_level,
        "absence_alert": bool(absence_row["absence_alert"]),
        "longest_silence_hours": absence_row["longest_tier_a_silence_hours"],
        "anomaly_score": round(float(anomaly_score), 3),
        "notable_changes": notable,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="path to the CASAS data file")
    parser.add_argument("--home", default="home", help="name used in the printout")
    parser.add_argument("--no-llm", action="store_true", help="skip Ollama, use the template")
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--compare-naive", action="store_true")
    parser.add_argument("--no-beacon", action="store_true", help="ignore dog tag data")
    arguments = parser.parse_args()

    print(f"loading {arguments.home} ...")
    events, sensor_zone_map = load_home(arguments.data)
    events = label_pet_events(events)
    if not arguments.no_beacon:
        events = resolve_night_motion(events)

    zone_counts = pd.Series(sensor_zone_map).value_counts().to_dict()
    print(f"  events: {len(events):,}")
    print(f"  days:   {events['date'].nunique()}")
    print(f"  zones inferred: {zone_counts}")
    print(f"  pet candidate motion events: {int(events['is_pet_candidate'].sum()):,}")
    if "is_dog_confirmed" in events.columns and events["is_dog_confirmed"].any():
        print(f"  beacon confirmed dog events:  {int(events['is_dog_confirmed'].sum()):,}")

    pet_frame = build_pet_classifier_frame(events)
    if len(pet_frame):
        pet_share = pet_frame["label_is_pet"].mean()
        print(f"  pet classifier training rows: {len(pet_frame):,} ({pet_share:.1%} pet)")

    daily = extract_daily_features(events, drop_pet_motion=True, use_beacon=not arguments.no_beacon)
    print(f"  daily feature rows: {len(daily)}")

    baseline = PersonalBaseline()
    deviations = baseline.transform(daily)
    daily.to_csv('daily_features_cache.csv', index=False)
    acute = AcuteDetector().fit(deviations)

    joblib.dump(acute.model, 'models/isolation.pkl')
    joblib.dump(baseline, 'models/baseline.pkl')  
    anomaly_scores = acute.score(deviations)

    watchdog = AbsenceWatchdog().fit(events)
    absence = watchdog.check(events)
    absence = absence[absence["date"].isin(daily["date"])].reset_index(drop=True)

    trend = TrendAnalyser().analyse(daily)

    print("\ntop trends over the last 28 days")
    print(trend.head(4).to_string(index=False))

    alert_levels = []
    for position in range(len(daily)):
        summary = summarise_alert(
            daily.iloc[position],
            absence.iloc[position],
            trend,
            anomaly_scores[position],
        )
        alert_levels.append(summary["alert_level"])

    level_counts = pd.Series(alert_levels).value_counts().to_dict()
    print(f"\nalert levels across {len(daily)} days: {level_counts}")

    if arguments.compare_naive:
        naive_daily = extract_daily_features(events, drop_pet_motion=False)
        naive_scores = SimpleThresholdDetector().score(naive_daily)
        naive_alerts = int((naive_scores > 0).sum())
        ours_alerts = int((anomaly_scores > 0).sum())
        print("\nnaive fixed-threshold system vs personalised system")
        print(f"  naive alerts:        {naive_alerts} over {len(naive_daily)} days")
        print(f"  personalised alerts: {ours_alerts} over {len(daily)} days")

    # Write the narrative for the most anomalous day, which is what you demo.
    worst_position = int(anomaly_scores.argmax())
    worst_summary = summarise_alert(
        daily.iloc[worst_position],
        absence.iloc[worst_position],
        trend,
        anomaly_scores[worst_position],
    )

    message, source = write_update(
        worst_summary, model_name=arguments.model, use_llm=not arguments.no_llm
    )

    print(f"\nmost anomalous day: {worst_summary['date']}")
    print(json.dumps(worst_summary, indent=2, default=str))
    print(f"\ncaregiver message (via {source}):")
    print(f"  {message}")


if __name__ == "__main__":
    main()