"""Writes a CASAS-format file so the pipeline can be tested without a download."""
import random
from datetime import datetime, timedelta

rng = random.Random(11)
ZONE_SENSORS = {"bedroom": ["M001","M002"], "bathroom": ["M003","M004"],
                "kitchen": ["M005","M006"], "living": ["M007","M008"]}
lines = []
def emit(ts, sensor, msg, act=None, marker=None):
    base = f"{ts.strftime('%Y-%m-%d')}\t{ts.strftime('%H:%M:%S.%f')}\t{sensor}\t{msg}"
    if act: base += f"\t{act}\t{marker}"
    lines.append(base)

start = datetime(2010, 1, 1)
for d in range(100):
    day = start + timedelta(days=d)
    decline = max(0.0, (d - 60) / 40.0)
    # night sleep block with dog wandering
    emit(day + timedelta(hours=0, minutes=5), "M001", "ON", "Sleeping", "begin")
    for _ in range(rng.randint(3, 9)):
        t = day + timedelta(hours=rng.uniform(0.5, 6))
        emit(t, rng.choice(ZONE_SENSORS[rng.choice(["kitchen","living"])]), "ON")
    trips = rng.randint(0, 1) + int(decline * 3)
    for _ in range(trips):
        t = day + timedelta(hours=rng.uniform(1, 5))
        emit(t, "M003", "ON", "Bed_to_toilet", "begin")
        emit(t + timedelta(minutes=4), "M003", "OFF", "Bed_to_toilet", "end")
    emit(day + timedelta(hours=6.5 + rng.uniform(0,1)), "M001", "OFF", "Sleeping", "end")
    rise = day + timedelta(hours=7 + decline * 1.5 + rng.uniform(0, 0.6))
    emit(rise, "M003", "ON", "Bathing", "begin")
    emit(rise + timedelta(minutes=12), "M003", "OFF", "Bathing", "end")
    meals = max(1, 4 - int(decline * 3))
    for m in range(meals):
        t = day + timedelta(hours=8 + m * 4 + rng.uniform(0, 1))
        emit(t, "M005", "ON", "Kitchen_Activity", "begin")
        emit(t + timedelta(minutes=20), "M005", "OFF", "Kitchen_Activity", "end")
        emit(t + timedelta(minutes=22), "M007", "ON", "Relax", "begin")
        emit(t + timedelta(minutes=90), "M007", "OFF", "Relax", "end")
    if rng.random() < 0.5 - decline * 0.4:
        t = day + timedelta(hours=10)
        emit(t, "D001", "OPEN", "Leave_Home", "begin")
        emit(t + timedelta(hours=1), "D001", "CLOSE", "Leave_Home", "end")
    for h in range(0, 24, 6):
        emit(day + timedelta(hours=h), "T001", f"{20 + rng.uniform(-2,2):.1f}")

lines.sort()
open("fixture_data", "w").write("\n".join(lines) + "\n")
print("fixture lines:", len(lines))
