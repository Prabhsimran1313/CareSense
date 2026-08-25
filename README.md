# CareSense: AI-Powered Older Adult Safety & Wellbeing Platform

CareSense is an ambient monitoring and behavioural analytics platform designed to support older adults living independently.

The current prototype uses public **CASAS smart-home sensor datasets** to identify unusual daily activity, prolonged inactivity, and gradual behavioural changes. The pipeline is also designed to support future **LoRaWAN-based sensor integration**.

CareSense combines personalised behavioural baselines, anomaly detection, custom inactivity monitoring, trend analysis, pet-aware activity processing, and optional local LLM-generated caregiver messages.

## Key Features

* Ambient smart-home sensor data processing
* Personalised behavioural baselines
* Unsupervised anomaly detection using Isolation Forest
* Custom prolonged-inactivity monitoring
* Long-term behavioural trend analysis
* Pet-aware motion filtering
* Local LLM-generated caregiver summaries
* Designed for future LoRaWAN sensor integration
* Extensible toward cognitive wellbeing and game-based features

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

## How It Works

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

## Dataset

The current prototype uses public **CASAS smart-home datasets**.

Two homes are particularly useful for experimentation:

* **Aruba:** single-resident smart-home dataset
* **Milan:** single-resident smart-home dataset that also includes a dog

The Milan dataset provides a useful scenario for exploring how pet-generated motion can affect naive activity-monitoring systems.

Each dataset contains timestamped events from sensors such as motion, door, temperature, and other household devices.

## Raw Sensor Processing

The first stage converts raw CASAS events into structured records containing information such as:

* Timestamp
* Sensor ID
* Sensor value
* Sensor type
* Activity annotation
* Functional room or zone
* Evidence tier

Sensor prefixes are used to identify categories such as motion, door, temperature, item interaction, light, beacon, and bed-related activity.

CASAS activity annotations can also be used to infer the likely room or functional zone associated with a sensor.

For future unlabelled LoRaWAN data, the sensor-to-room mapping would need to be provided during device configuration.

## Behavioural Feature Engineering

Raw sensor events are not passed directly into the machine-learning model.

Instead, the event stream is converted into one behavioural feature vector per day.

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

## Pet-Aware Activity Processing

Simple motion sensors can be triggered by pets, which may create false activity signals.

CareSense therefore distinguishes stronger evidence of human activity from general motion using evidence tiers.

For the CASAS prototype, contextual activity labels are used to identify likely pet-generated motion in single-resident homes. For example, motion elsewhere in the home while the resident is annotated as sleeping can be treated as a possible pet-generated event.

The future hardware design can also support BLE beacon-based pet identification.

## Personalised Baseline

Different residents naturally have different routines, so CareSense does not rely on one global behavioural threshold.

`PersonalBaseline` compares each resident against their own recent history using:

* approximately a **28-day rolling window**
* rolling median
* Median Absolute Deviation (MAD)
* a minimum history of approximately 10 days

The current day is excluded from its own baseline calculation.

Each behavioural feature is converted into a personalised deviation before being passed to downstream detectors.

This allows the system to ask:

> Is today's behaviour unusual for this resident?

rather than:

> Is today's behaviour unusual for everyone?

## Detection Models

CareSense uses three complementary detection mechanisms.

| Component            | Purpose                                       |
| -------------------- | --------------------------------------------- |
| **Isolation Forest** | Detect unusual daily behaviour                |
| **Absence Watchdog** | Detect prolonged low confirmed human activity |
| **Trend Analyser**   | Detect gradual behavioural change             |

### Isolation Forest

Isolation Forest is used for unsupervised anomaly detection because CASAS does not contain enough labelled emergency or deterioration events for reliable supervised classification.

The model operates on personalised behavioural deviations rather than raw sensor counts and identifies days where several behaviours differ significantly from the resident's normal routine.

The current configuration uses:

```python
IsolationForest(
    n_estimators=300,
    contamination=0.03,
    random_state=7
)
```

### Absence Watchdog

`AbsenceWatchdog` is a custom deterministic safety component implemented in the project.

It learns personalised thresholds from historical Tier A activity and generates an alert when:

1. confirmed human activity is unusually low, and
2. the silent period is unusually long.

This component is kept separate from Isolation Forest because prolonged inactivity is safety-sensitive and benefits from deterministic, testable logic.

### Trend Analyser

`TrendAnalyser` uses:

* Theil-Sen Regression
* Custom Change-Point Detection

Theil-Sen estimates whether behavioural features are gradually increasing or decreasing over time and is less sensitive to occasional unusual days than ordinary linear regression.

The custom change-point detector attempts to identify when a meaningful behavioural shift began.

## Alert Levels

Detector outputs are combined into simple alert levels:

```text
Absence alert
    ↓
Urgent

Acute anomaly or notable trend
    ↓
Nudge

No meaningful change
    ↓
Silent
```

The alert decision is made before the LLM is called.

## Local LLM Integration

CareSense optionally uses a pretrained local language model through **Ollama** to convert detector outputs into caregiver-friendly summaries.

The project does **not** train its own LLM.

The LLM only receives a small structured summary of detector results. It does not receive the raw sensor event stream and does not decide whether an alert should be generated.

```text
Sensor Data
    ↓
Analytics + Detection
    ↓
Alert Level + Summary
    ↓
Local LLM
    ↓
Caregiver-Friendly Message
```

If Ollama is unavailable, `narrative.py` falls back to a deterministic text template so the monitoring pipeline does not depend on the LLM service.

Supported local models can include:

* Llama 3.1 8B
* Qwen 2.5 7B
* Phi-3 Mini

## Project Structure

```text
casas_loader.py   Parses CASAS events, identifies sensor types,
                  infers room zones, and assigns evidence tiers.

features.py       Filters likely pet-related motion and converts
                  raw events into daily behavioural features.

detectors.py      Implements PersonalBaseline, Isolation Forest-based
                  AcuteDetector, AbsenceWatchdog, and TrendAnalyser.

narrative.py      Converts detector outputs into caregiver-friendly
                  messages using a local LLM through Ollama.

run_pipeline.py   Orchestrates the complete analytics pipeline.
```

## Setup

Install the core Python dependencies:

```bash
pip install pandas numpy scikit-learn
```

The custom change-point detector is implemented directly in `detectors.py`, so an additional library such as `ruptures` is not required.

Ollama is optional and is only required for local LLM-generated caregiver messages.

## Get the Data

Download the Aruba and Milan datasets from CASAS.

Create the required directory structure:

```bash
mkdir -p data
```

Place the files as follows:

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

To enable LLM-generated caregiver messages, configure Ollama and remove the `--no-llm` flag.

### Optional Ollama Setup

```bash
brew install ollama
ollama serve
ollama pull llama3.1:8b
```

Alternative models:

```bash
ollama pull qwen2.5:7b
ollama pull phi3:mini
```

## Personalised vs Naive Alerting

The repository includes a naive comparison detector to evaluate:

```text
Global / naive thresholds
vs
Personalised behavioural monitoring
```

This is particularly useful when comparing Aruba and Milan because Milan includes a dog and may therefore be more affected by pet-generated motion.

The personalised approach combines:

* resident-specific behavioural baselines
* evidence tiers
* pet-aware processing
* anomaly detection

to reduce unnecessary alerts.

## Known Limitations

1. **Zone inference depends on activity labels:** CASAS annotations are used to infer room zones. Future unlabelled LoRaWAN data would require a sensor-to-room mapping.

2. **Pet inference assumes single occupancy:** the current contextual pet logic does not directly generalise to multi-resident homes.

3. **CASAS does not contain labelled falls or confirmed deterioration events:** it is useful for validating sensor processing, feature extraction, and pet-aware logic, while synthetic scenarios are used to explore gradual decline behaviour.

4. **The system does not provide medical diagnosis:** CareSense identifies behavioural changes and unusual activity patterns as supportive information only.

## Future Development

Potential extensions include:

* Direct LoRaWAN sensor ingestion
* Real-time caregiver dashboards and alerts
* Improved pet identification
* Multi-resident support
* Cognitive games for memory, attention, and problem-solving
* Longitudinal cognitive wellbeing indicators
* Personalised activity summaries

## Project Direction

The long-term goal of CareSense is to combine passive ambient monitoring with cognitive and wellbeing features to create a broader support platform for older adults living independently.

The platform is designed to provide useful behavioural insights while keeping safety decisions personalised, explainable, and independent of generative AI.
