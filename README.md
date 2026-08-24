<<<<<<< HEAD
# CareSense – AI-Powered Older Adult Safety & Wellbeing Platform

Detects acute emergencies and slow deterioration for someone living alone,
from binary ambient sensors. Runs on CASAS data now, on your LoRaWAN nodes later.

## Setup

```bash
pip install pandas numpy scikit-learn
```

That is the whole dependency list. The change point detector is written out in
`detectors.py` rather than pulled from `ruptures`, so there is nothing else to install.

## Get the data

Download Aruba and Milan from CASAS. Each home is a single text file, usually
called `data`, with one event per line.

```bash
mkdir -p data
# put the Aruba file at data/aruba/data and the Milan file at data/milan/data
```

## Run it

```bash
python run_pipeline.py --data data/aruba/data --home aruba --no-llm
python run_pipeline.py --data data/milan/data --home milan --no-llm --compare-naive
```

Drop `--no-llm` once Ollama is running to get the generated caregiver message.

## The experiment that matters

Aruba is a single resident. Milan is a single resident **with a dog**. Same
sensor types, same task, one variable.

```bash
python run_pipeline.py --data data/aruba/data --home aruba --no-llm --compare-naive
python run_pipeline.py --data data/milan/data --home milan --no-llm --compare-naive
```

Compare the naive alert count against the personalised one on each. The gap
should widen on Milan. That is your results slide, on real public data.

## How it fits together

```
casas_loader.py   parse events, infer room zones from activity labels,
                  assign evidence tiers (A = pet cannot fake it)
features.py       label pet events for free, collapse to one row per day
detectors.py      PersonalBaseline -> AcuteDetector (Isolation Forest)
                  AbsenceWatchdog (the one that catches the dangerous case)
                  TrendAnalyser (Theil-Sen slope + change point)
narrative.py      local LLM turns the numbers into a sentence
run_pipeline.py   wires it together
```

Two design rules worth keeping:

**The LLM never decides whether to alert.** That happens in `detectors.py`,
deterministically, so it is testable and tunable. The model only decides how to
say it, and it only ever sees a handful of numbers — never the raw event stream.

**Nothing downstream sees a raw count.** `PersonalBaseline` converts every
feature into a deviation from that person's own 28-day rolling median. That is
what makes one global model behave personally, and it works from day ten rather
than needing months of history.

## Local LLM

```bash
brew install ollama
ollama serve
ollama pull llama3.1:8b     # good on an M-series Mac
ollama pull qwen2.5:7b      # lighter
ollama pull phi3:mini       # if memory is tight
```

`narrative.py` falls back to a template if Ollama is not reachable. Keep that
fallback — a demo that dies because a model server was not started is a bad
afternoon, and nobody watching can tell which one wrote the text.

## Tuning

- `AbsenceWatchdog(count_fraction=..., silence_percentile=...)` — raise
  `count_fraction` for more sensitivity. Both thresholds are learned per home,
  because a house with one door sensor and a house with four cannot share a number.
- `AcuteDetector(contamination=...)` — your alert rate, roughly.
- `PersonalBaseline(window_days=...)` — shorter reacts faster, longer is steadier.
- `TrendAnalyser(change_point_threshold=...)` — how strong a shift must be
  before it gets a date attached.

## Known limitations — say these out loud in the pitch

1. **Zone inference needs activity labels.** It learns which sensor is in which
   room from the annotations. On unlabelled data you must supply a map by hand.
2. **The pet labeller assumes single occupancy.** Motion elsewhere during
   annotated sleep is a pet only if nobody else lives there. It is wrong on Cairo.
3. **`notable_changes` is computed over the whole run**, so it attaches the same
   trend to every day. For a live system, recompute per day over a trailing window.
4. **CASAS has no falls and no labelled deterioration.** Real data validates the
   activity and pet layers; the synthetic generator validates the decline layer.
   Be explicit about which is which — it is a stronger position than pretending
   one dataset does both.
5. **Bathroom motion stands in for a toilet flush sensor.** On your own hardware
   the piezo on the cistern is a true Tier A signal and this gets much sharper.
=======
# Caresense
AI-powered safety and wellbeing platform for older adults using ambient sensor analytics, anomaly detection, trend analysis, and cognitive wellbeing features.
>>>>>>> a8476623be62ff00d48ff5a7af682f9fef5000b4
