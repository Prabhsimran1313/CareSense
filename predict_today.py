import joblib
import pandas as pd
from features import DAILY_FEATURE_NAMES
from narrative import write_update


def predict_day(day_features_dict, model_name="llama3:8b", use_llm=True):
    """
    day_features_dict: one day's values, e.g.
    {'night_bathroom_trips': 3, 'first_rise_hour': 9.5, 'kitchen_visits': 1,
     'median_transit_seconds': 45, 'hours_active': 6, 'longest_inactive_minutes': 240,
     'door_openings': 0, 'sleep_fragmentation': 5, 'tier_a_event_count': 2}
    """
    baseline = joblib.load('models/baseline.pkl')
    model = joblib.load('models/isolation.pkl')

    today = pd.DataFrame([day_features_dict])
    today['date'] = pd.Timestamp.now().normalize()

    history = pd.read_csv('daily_features_cache.csv', parse_dates=['date'])
    combined = pd.concat([history, today], ignore_index=True)
    deviations = baseline.transform(combined)

    X = deviations[DAILY_FEATURE_NAMES].tail(1)
    is_anomaly = model.predict(X)[0] == -1
    score = float(model.decision_function(X)[0])

    # --- turn the raw model output into an actual alert ---
    tier_a_count = day_features_dict.get("tier_a_event_count", 0)
    silence_hours = day_features_dict.get("longest_inactive_minutes", 0) / 60.0

    if tier_a_count < 2 or silence_hours >= 9.0:
        alert_level = "urgent"
    elif is_anomaly:
        alert_level = "nudge"
    else:
        alert_level = "silent"

    alert_summary = {
        "alert_level": alert_level,
        "absence_alert": alert_level == "urgent",
        "longest_silence_hours": round(silence_hours, 1),
        "anomaly_score": round(score, 3),
        "notable_changes": [],
    }

    message, source = write_update(alert_summary, model_name=model_name, use_llm=use_llm)

    return {
        "alert_level": alert_level,
        "is_anomaly": is_anomaly,
        "score": round(score, 3),
        "message": message,
        "message_source": source,
    }


if __name__ == "__main__":
    result = predict_day({
        "night_bathroom_trips": 4,
        "first_rise_hour": 11.0,
        "kitchen_visits": 0,
        "median_transit_seconds": 45,
        "hours_active": 1,
        "longest_inactive_minutes": 560,
        "door_openings": 0,
        "sleep_fragmentation": 5,
        "tier_a_event_count": 1,
    }, model_name="llama3:8b")

    print(f"ALERT LEVEL: {result['alert_level'].upper()}")
    print(f"anomaly score: {result['score']}  (source: {result['message_source']})")
    print(f"\nmessage:\n  {result['message']}")