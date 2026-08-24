# CareSense – AI-Powered Older Adult Safety & Wellbeing Platform

CareSense is an ambient monitoring and behavioural analytics platform designed to support older adults living independently.

The current prototype uses public CASAS smart-home sensor datasets to identify unusual daily activity, prolonged inactivity, and gradual behavioural changes. The same pipeline is designed to support future LoRaWAN-based sensor data.

The system combines personalised behavioural baselines, anomaly detection, trend analysis, pet-aware activity filtering, and optional local LLM-generated caregiver messages.

## Key Features

* Ambient smart-home sensor data processing
* Personalised 28-day behavioural baselines
* Unsupervised anomaly detection using Isolation Forest
* Custom prolonged-inactivity monitoring
* Long-term behavioural trend analysis
* Pet-aware motion filtering
* Caregiver-friendly summaries using a local LLM
* Designed for future LoRaWAN sensor integration
* Extensible toward cognitive wellbeing and game-based monitoring features

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Isolation Forest
* Theil-Sen Regression
* Custom Change-Point Detection
* CASAS Smart-Home Dataset
* Ollama
* Llama 3.1 / Qwen / Phi models
* Future LoRaWAN sensor integration

## How the System Works

```text
CASAS Dataset / Future LoRaWAN Sensors
                ↓
         Raw Sensor Events
                ↓
         Event Processing
                ↓
     Behavioural Feature Engineering
                ↓
       Personalised Baseline
                ↓
 ┌──────────────┼──────────────┐
 ↓              ↓              ↓
Isolation    Absence        Trend
Forest       Watchdog       Analyser
 ↓              ↓              ↓
 └──────────────┼──────────────┘
                ↓
          Alert Decision
                ↓
        Local LLM via Ollama
                ↓
     Caregiver-Friendly Message
```

## Project Structure

```text
casas_loader.py   Parses CASAS events, identifies sensor types,
                  infers room zones, and assigns evidence tiers.

features.py       Filters likely pet-related motion and converts
                  raw sensor events into daily behavioural features.

detectors.py      PersonalBaseline:
                  Creates personalised behavioural deviations.

                  AcuteDetector:
                  Uses Isolation Forest for anomaly detection.

                  AbsenceWatchdog:
                  Detects unusually low confirmed human activity
                  combined with prolonged silence.

                  TrendAnalyser:
                  Uses Theil-Sen regression and custom change-point
                  detection to identify longer-term behavioural changes.

narrative.py      Converts detector outputs into caregiver-friendly
                  messages using a local LLM through Ollama.

run_pipeline.py   Orchestrates the complete analytics pipeline.
```

## Dataset

The current prototype uses the CASAS smart-home datasets.

Two homes are particularly useful for experimentation:

* **Aruba** – single-resident smart-home dataset
* **Milan** – single-resident smart-home dataset that also includes a dog

The Milan dataset provides a useful scenario for exploring how pet-generated motion can affect naive activity-monitoring systems.

Each dataset contains timestamped sensor events such as motion, door activity, temperature, and other household interactions.

## Setup

Install the core Python dependencies:

```bash
pip install pandas numpy scikit-learn
```

These are the main dependencies required for the analytics pipeline.

The custom change-point detector is implemented directly in `detectors.py`, so an additional change-point library such as `ruptures` is not required.

Ollama is optional and is only required if local LLM-generated caregiver messages are enabled.

## Get the Data

Download the Aruba and Milan datasets from CASAS.

Each home is provided as a text-based sensor event file, typically named `data`.

Create the required directory structure:

```bash
mkdir -p data
```

Then place the files as follows:

```text
data/
├── aruba/
│   └── data
└── milan/
    └── data
```

## Run the Pipeline

Run the Aruba dataset:

```bash
python run_pipeline.py --data data/aruba/data --home aruba --no-llm
```

Run the Milan dataset and compare the personalised approach with the naive detector:

```bash
python run_pipeline.py --data data/milan/data --home milan --no-llm --compare-naive
```

To enable caregiver messages generated through the local LLM, remove the `--no-llm` flag after Ollama has been configured.

## Raw Sensor Processing

The first stage of the pipeline converts raw CASAS sensor events into structured data.

For each event, the system processes information such as:

* Timestamp
* Sensor ID
* Sensor value
* Sensor type
* Activity annotation
* Functional room or zone
* Evidence tier

Sensor prefixes are used to identify different sensor categories such as motion, door, temperature, item interaction, light, beacon, and bed-related activity.

Room zones can be inferred from CASAS activity annotations.

For future unlabelled LoRaWAN data, the sensor-to-room mapping would need to be provided during device configuration.

## Evidence Tiers

Not every sensor event provides the same confidence that a person was actually active.

Simple motion may be triggered by a pet, while other interactions can provide stronger evidence of human activity.

CareSense therefore uses evidence tiers to distinguish stronger human-activity signals from general motion.

This is especially important for prolonged-inactivity monitoring.

## Pet-Aware Activity Processing

Pet-generated motion can create false activity signals in smart-home monitoring.

The prototype uses contextual information from the CASAS dataset to identify likely pet-related motion in single-resident homes.

For example, motion detected in another room while the resident is annotated as sleeping can be treated as a possible pet-generated event.

The future hardware design can also support BLE beacon-based pet identification.

This pet-filtering approach is designed to reduce false activity without removing stronger human-activity evidence.

## Behavioural Feature Engineering

Raw sensor events are not passed directly into the machine-learning model.

Instead, thousands of sensor events are converted into one behavioural feature vector per day.

The current feature set includes:

* `night_bathroom_trips`
* `first_rise_hour`
* `kitchen_visits`
* `median_transit_seconds`
* `hours_active`
* `longest_inactive_minutes`
* `door_openings`
* `sleep_fragmentation`
* `tier_a_event_count`

These features provide a compact representation of the resident's daily routine.

## Personalised Baseline

Different people naturally have different routines.

For example, one resident may normally wake at 6:30 AM while another normally wakes at 9:00 AM.

For this reason, CareSense does not rely on one global behavioural threshold.

`PersonalBaseline` compares each resident against their own recent history.

The current implementation uses:

* approximately a **28-day rolling window**
* rolling median
* Median Absolute Deviation (MAD)
* a minimum history of approximately 10 days

The current day is excluded from its own baseline calculation.

Each behavioural feature is converted into a personalised deviation before being passed downstream.

This allows the system to ask:

> Is today's behaviour unusual for this resident?

rather than:

> Is today's behaviour unusual for everyone?

## Acute Anomaly Detection

The main anomaly-detection model is Isolation Forest.

Isolation Forest is used because the available CASAS data does not contain enough labelled emergency or deterioration events to train a supervised classifier reliably.

The model receives personalised behavioural deviations rather than raw sensor counts.

The current configuration uses approximately:

```python
IsolationForest(
    n_estimators=300,
    contamination=0.03,
    random_state=7
)
```

The model evaluates the combination of behavioural features and identifies days that are significantly different from the resident's normal pattern.

For example, a day may become anomalous if several changes occur together:

* much later first activity
* unusually low daily activity
* fewer kitchen visits
* prolonged inactivity
* increased nighttime movement

Isolation Forest identifies the day as unusual but does not make a medical diagnosis.

## Absence Watchdog

`AbsenceWatchdog` is a custom safety component implemented in the project.

It is not a pretrained machine-learning model.

Its purpose is to identify potentially concerning periods where there is:

1. unusually low confirmed human activity, and
2. an unusually long silent period.

The Watchdog learns personalised thresholds from historical Tier A activity.

It considers factors such as:

* typical number of strong human-activity events
* normal inactivity duration
* unusually low activity counts
* unusually long silent periods

Both conditions must be sufficiently unusual before an absence alert is generated.

This component is kept separate from Isolation Forest because prolonged inactivity is safety-sensitive and benefits from deterministic, testable logic.

## Trend Analysis

CareSense also looks for gradual behavioural changes that may not appear as a single-day anomaly.

`TrendAnalyser` uses:

* Theil-Sen Regression
* Custom Change-Point Detection

Theil-Sen is used because it is robust to occasional unusual days.

It estimates whether behavioural features are gradually increasing or decreasing over time.

Examples include:

* declining daily activity
* increasing nighttime bathroom activity
* decreasing kitchen visits
* changing room-transition timing

The custom change-point detector attempts to identify when a meaningful shift in behaviour began.

This gives the system three complementary detection mechanisms:

| Component        | Purpose                                 |
| ---------------- | --------------------------------------- |
| Isolation Forest | Detect unusual daily behaviour          |
| Absence Watchdog | Detect prolonged low confirmed activity |
| Trend Analyser   | Detect gradual behavioural change       |

## Alert Levels

Detector outputs are combined into simple alert levels.

Conceptually:

```text
Absence alert
    ↓
Urgent

Acute anomaly or notable behavioural trend
    ↓
Nudge

No meaningful change
    ↓
Silent
```

The alert decision is made before the LLM is called.

## Local LLM Integration

CareSense optionally uses a local language model through Ollama to convert technical detector outputs into simple caregiver-friendly messages.

The project does **not** train its own LLM.

Instead, Ollama is used to run existing pretrained models locally, such as:

* Llama 3.1 8B
* Qwen 2.5 7B
* Phi-3 Mini

Install Ollama on macOS:

```bash
brew install ollama
```

Start the local server:

```bash
ollama serve
```

Then pull a model:

```bash
ollama pull llama3.1:8b
```

Alternative lighter models can also be used:

```bash
ollama pull qwen2.5:7b
ollama pull phi3:mini
```

## LLM Safety Design

A key design principle is:

**The LLM never decides whether an alert should be generated.**

Detection is handled by the analytics and detector components.

The LLM only receives a small structured summary of the results and converts it into natural language.

It does not receive the full raw sensor event stream.

Conceptually:

```text
Sensor Data
    ↓
Analytics + Detection
    ↓
Alert level and summary
    ↓
Local LLM
    ↓
Caregiver-friendly explanation
```

This keeps the alert logic deterministic, testable, and independent of generative model behaviour.

## LLM Fallback

If Ollama or the selected local model is unavailable, `narrative.py` falls back to a deterministic text template.

This ensures that the monitoring pipeline does not depend on the LLM service being available.

## Personalised vs Naive Alerting

The repository includes a naive comparison detector.

This allows the project to compare:

```text
Global / naive thresholds
vs
Personalised behavioural monitoring
```

The experiment is particularly useful when comparing Aruba and Milan.

Because Milan includes a dog, motion-based naive detection may be more affected by pet-generated activity.

The personalised pipeline combines:

* individual behavioural baselines
* evidence tiers
* pet-aware processing
* anomaly detection

to reduce unnecessary alerts.

## Tuning

Several detector settings can be adjusted.

### Absence Watchdog

```text
AbsenceWatchdog(
    count_fraction=...,
    silence_percentile=...
)
```

* Increasing `count_fraction` makes the activity-count rule more sensitive.
* `silence_percentile` controls how unusual a silent period must be.
* Thresholds are learned separately for each home.

### Acute Detector

```text
AcuteDetector(contamination=...)
```

The contamination value influences the expected anomaly rate.

### Personal Baseline

```text
PersonalBaseline(window_days=...)
```

A shorter baseline adapts more quickly.

A longer baseline is generally more stable.

### Trend Analyser

```text
TrendAnalyser(change_point_threshold=...)
```

This controls how strong a behavioural shift must be before it is treated as a meaningful change point.

## Known Limitations

1. **Zone inference depends on activity labels.**
   CASAS activity annotations are used to infer which sensors belong to which room or zone. With unlabelled LoRaWAN data, the sensor-to-room mapping would need to be supplied during installation.

2. **Pet inference currently assumes single occupancy.**
   Motion detected elsewhere while the resident is annotated as sleeping can be interpreted as pet activity only when one person lives in the home. This approach does not directly generalise to multi-resident environments.

3. **Trend results should be recomputed using a trailing window in a live deployment.**
   The current prototype calculates notable changes over the experimental run. A production system should calculate trends using only data available up to the current day.

4. **CASAS does not contain labelled falls or confirmed deterioration events.**
   CASAS is useful for validating sensor processing, behavioural feature extraction, and pet-aware logic. Synthetic scenarios are used to explore gradual decline behaviour. Real-world deployment would require longitudinal validation using appropriately labelled data.

5. **Bathroom motion is currently used as a behavioural proxy.**
   The CASAS datasets do not provide the same dedicated bathroom sensing planned for future hardware. A dedicated sensor could provide stronger evidence of specific interactions.

6. **The system does not provide medical diagnosis.**
   CareSense identifies behavioural changes and unusual activity patterns. Any generated insight should be treated as supportive information rather than a clinical conclusion.

## Future Development

The broader CareSense concept is designed to extend beyond ambient monitoring.

Potential future modules include:

* Direct LoRaWAN sensor ingestion
* Real-time caregiver dashboards
* Mobile alerts
* Improved pet identification
* Multi-resident support
* Cognitive games for memory, attention, and problem-solving
* Longitudinal cognitive wellbeing indicators
* Personalised activity summaries
* Caregiver and family communication tools
* Integration with additional support services

## Project Goal

The long-term goal of CareSense is to combine passive ambient monitoring with cognitive and wellbeing features to create a broader support platform for older adults living independently.

The platform is designed to provide useful behavioural insights while keeping safety decisions explainable, personalised, and independent of generative AI.
