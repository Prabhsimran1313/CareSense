"""
The four models.

  1. PersonalBaseline    - rolling median and MAD per feature, per person
  2. AcuteDetector       - Isolation Forest on personalised deviation scores
  3. AbsenceWatchdog     - expected Tier A events that never happened
  4. TrendAnalyser       - Theil-Sen slope plus a change point search

Nothing here ever sees a raw count. Everything works on deviations from the
person's own history, which is what makes one global model behave personally.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import TheilSenRegressor

from features import DAILY_FEATURE_NAMES


class PersonalBaseline:
    """
    Rolling median and median absolute deviation for each feature.

    Median and MAD are used instead of mean and standard deviation so that one
    bad night does not drag the baseline along with it.
    """

    def __init__(self, window_days=28, minimum_days=10):
        self.window_days = window_days
        self.minimum_days = minimum_days

    def transform(self, daily_features):
        deviations = pd.DataFrame({"date": daily_features["date"]})

        for feature_name in DAILY_FEATURE_NAMES:
            series = daily_features[feature_name].astype(float)

            # Shift by one so today is never part of its own baseline.
            rolling = series.shift(1).rolling(self.window_days, min_periods=self.minimum_days)
            rolling_median = rolling.median()
            rolling_mad = rolling.apply(
                lambda window: np.median(np.abs(window - np.median(window))), raw=True
            )

            # 1.4826 scales MAD so it is comparable to a standard deviation.
            scaled_mad = (rolling_mad * 1.4826).replace(0, np.nan)
            fallback = series.std() if series.std() > 0 else 1.0
            scaled_mad = scaled_mad.fillna(fallback)

            deviations[feature_name] = (series - rolling_median) / scaled_mad

        # Early days have no baseline yet, so treat them as perfectly normal.
        return deviations.fillna(0.0)


class AcuteDetector:
    """Isolation Forest over the personalised deviation vectors."""

    def __init__(self, contamination=0.03, random_state=7):
        self.model = IsolationForest(
            n_estimators=300,
            contamination=contamination,
            random_state=random_state,
        )
        self.is_fitted = False

    def fit(self, deviations):
        self.model.fit(deviations[DAILY_FEATURE_NAMES].values)
        self.is_fitted = True
        return self

    def score(self, deviations):
        """Higher score means more anomalous. Zero is the alert threshold."""
        raw = self.model.decision_function(deviations[DAILY_FEATURE_NAMES].values)
        return -raw


class SimpleThresholdDetector:
    """
    The naive comparison model, on purpose.

    Fixed global thresholds with no personalisation. Every serious system should
    have to beat this, and showing the gap is the strongest slide you can build.
    """

    def __init__(self, z_limit=2.0):
        self.z_limit = z_limit

    def score(self, daily_features):
        values = daily_features[DAILY_FEATURE_NAMES].astype(float)
        population_z = (values - values.mean()) / values.std().replace(0, 1.0)
        return population_z.abs().max(axis=1).values - self.z_limit


class AbsenceWatchdog:
    """
    Watches for expected Tier A events that never arrived.

    This is the detector that catches the dangerous case. A pet keeps motion
    sensors busy all day, so a motion based system sees a normal day while the
    resident is unconscious. Tier A events cannot be faked by an animal.
    """

    def __init__(self, waking_hours=(8, 20), count_fraction=0.34, silence_percentile=97.5):
        self.waking_hours = waking_hours
        self.count_fraction = count_fraction
        self.silence_percentile = silence_percentile
        self.expected_daily_count = None
        self.minimum_expected_events = None
        self.silence_limit_hours = None

    def fit(self, events):
        """
        Learn what a quiet day looks like for this person specifically.

        Homes differ enormously in how many Tier A sensors they have. Aruba has
        four door sensors, another home might have one. A fixed threshold would
        alert every day in a sparsely instrumented house and never alert in a
        busy one, so both thresholds are learned from the person's own history.
        """
        per_day, silences = self._daily_stats(events)

        if len(per_day):
            self.expected_daily_count = float(per_day.median())
            self.minimum_expected_events = max(1.0, self.expected_daily_count * self.count_fraction)
        else:
            self.expected_daily_count = 0.0
            self.minimum_expected_events = 1.0

        window_hours = self.waking_hours[1] - self.waking_hours[0]
        if len(silences):
            learned_limit = float(np.percentile(silences, self.silence_percentile))
            # Never alert on a silence the person routinely has anyway.
            self.silence_limit_hours = min(window_hours, max(3.0, learned_limit))
        else:
            self.silence_limit_hours = window_hours
        return self

    def _daily_stats(self, events):
        """Tier A events per day and the longest silent stretch per day."""
        tier_a = events[events["tier"] == "A"].copy()
        tier_a["hour"] = tier_a["timestamp"].dt.hour
        in_window = tier_a[
            (tier_a["hour"] >= self.waking_hours[0]) & (tier_a["hour"] < self.waking_hours[1])
        ]

        counts, silences = [], []
        for day in events["date"].unique():
            day_events = in_window[in_window["date"] == day]
            counts.append(len(day_events))
            silences.append(self._longest_silence(day, day_events))
        return pd.Series(counts), np.array(silences)

    def _longest_silence(self, day, day_events):
        window_start = day + pd.Timedelta(hours=self.waking_hours[0])
        window_end = day + pd.Timedelta(hours=self.waking_hours[1])
        if len(day_events) == 0:
            return (window_end - window_start).total_seconds() / 3600.0
        stamps = day_events["timestamp"].sort_values()
        boundaries = pd.concat(
            [pd.Series([window_start]), stamps, pd.Series([window_end])]
        ).reset_index(drop=True)
        return boundaries.diff().max().total_seconds() / 3600.0

    def check(self, events):
        """Return one row per day flagging silent Tier A periods."""
        if self.minimum_expected_events is None:
            raise RuntimeError("call fit() before check()")

        tier_a = events[events["tier"] == "A"].copy()
        tier_a["hour"] = tier_a["timestamp"].dt.hour
        in_window = tier_a[
            (tier_a["hour"] >= self.waking_hours[0]) & (tier_a["hour"] < self.waking_hours[1])
        ]

        results = []
        for day in sorted(events["date"].unique()):
            day_events = in_window[in_window["date"] == day]
            count = len(day_events)
            longest_silence_hours = self._longest_silence(day, day_events)

            count_is_low = count < self.minimum_expected_events
            silence_is_long = longest_silence_hours >= self.silence_limit_hours

            results.append(
                {
                    "date": day,
                    "tier_a_count": count,
                    "longest_tier_a_silence_hours": round(longest_silence_hours, 2),
                    "absence_alert": int(count_is_low and silence_is_long),
                }
            )
        return pd.DataFrame(results)


def find_change_point(values, minimum_segment=10):
    """
    Binary segmentation change point search.

    Splits the series at the point that most reduces total squared error, and
    returns that index with the improvement it bought. Written out rather than
    imported so the project has no extra dependency.
    """
    values = np.asarray(values, dtype=float)
    total_points = len(values)
    if total_points < 2 * minimum_segment:
        return None, 0.0

    baseline_error = float(np.sum((values - values.mean()) ** 2))
    best_index, best_error = None, baseline_error

    for split_index in range(minimum_segment, total_points - minimum_segment):
        left, right = values[:split_index], values[split_index:]
        split_error = float(
            np.sum((left - left.mean()) ** 2) + np.sum((right - right.mean()) ** 2)
        )
        if split_error < best_error:
            best_error, best_index = split_error, split_index

    improvement = (baseline_error - best_error) / baseline_error if baseline_error > 0 else 0.0
    return best_index, improvement


class TrendAnalyser:
    """Robust slope estimation plus change point detection, per feature."""

    def __init__(self, window_days=28, change_point_threshold=0.25):
        self.window_days = window_days
        self.change_point_threshold = change_point_threshold

    def analyse(self, daily_features):
        recent = daily_features.tail(self.window_days).reset_index(drop=True)
        day_index = np.arange(len(recent)).reshape(-1, 1)

        findings = []
        for feature_name in DAILY_FEATURE_NAMES:
            series = recent[feature_name].astype(float).values
            if np.allclose(series, series[0]):
                continue

            estimator = TheilSenRegressor(random_state=7, max_subpopulation=2000)
            estimator.fit(day_index, series)
            slope_per_day = float(estimator.coef_[0])

            level = np.median(np.abs(series))
            percent_per_week = (slope_per_day * 7.0 / level * 100.0) if level > 0 else 0.0

            full_series = daily_features[feature_name].astype(float).values
            split_index, improvement = find_change_point(full_series)
            change_date = None
            if split_index is not None and improvement >= self.change_point_threshold:
                change_date = str(daily_features["date"].iloc[split_index].date())

            findings.append(
                {
                    "feature": feature_name,
                    "slope_per_day": round(slope_per_day, 4),
                    "percent_change_per_week": round(percent_per_week, 2),
                    "change_point_date": change_date,
                    "change_point_strength": round(improvement, 3),
                }
            )

        findings_frame = pd.DataFrame(findings)
        return findings_frame.reindex(
            findings_frame["percent_change_per_week"].abs().sort_values(ascending=False).index
        ).reset_index(drop=True)
