"""
FastAPI layer. All the real logic lives in logic.py and is tested there —
this file is just routing and request/response shapes.

Run it:
    pip install fastapi uvicorn pandas requests
    uvicorn api:app --reload --port 8000

Then, with simulator.py writing to the same sensor_log.csv in this folder:
    curl http://localhost:8000/status
    curl http://localhost:8000/rooms
    curl http://localhost:8000/trend
    curl http://localhost:8000/gaps
    curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
         -d '{"question": "what happened in the kitchen in the last 10 minutes"}'
"""

from typing import Optional

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from logic import (
    chunk_to_text,
    extract_relevant_chunk,
    find_room_gaps,
    get_activity_trend,
    get_current_status,
    get_naive_vs_personalised,
    get_room_activity,
    load_log,
)

app = FastAPI(title="Quiet Care API")

# Wide open for a hackathon demo. Lock this down to your app's origin
# before this ever runs against anything real.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3:8b"

ASK_SYSTEM_PROMPT = """You answer questions about a household sensor log for a caregiver.
You are given only a SMALL, ALREADY FILTERED slice of the log, not the full history.
Answer only from the events shown. If the events don't cover the question, say so plainly.
Keep it to two or three sentences. Do not diagnose. Do not invent numbers not in the data."""


# ---------------------------------------------------------------------------
# Chart endpoints
# ---------------------------------------------------------------------------

@app.get("/status")
def status():
    """Current alert state — for the top status card on screen 1."""
    return get_current_status(load_log())


@app.get("/rooms")
def rooms():
    """Last-seen time per room — for the room grid on screen 1."""
    return {"rooms": get_room_activity(load_log())}


@app.get("/trend")
def trend(bucket_minutes: int = 1, window_minutes: int = 30):
    """Bucketed event counts — for the sparkline / bar chart on screen 2."""
    return {"buckets": get_activity_trend(load_log(), bucket_minutes, window_minutes)}


@app.get("/comparison")
def comparison(window_minutes: int = 30):
    """Naive vs personalised alert count — the results chart on screen 2."""
    return get_naive_vs_personalised(load_log(), window_minutes)


@app.get("/gaps")
def gaps():
    """
    Rooms where the current stay has run longer than expected with no
    activity elsewhere in the house — e.g. bedroom for 10+ hours straight.
    """
    warnings = find_room_gaps(load_log())
    return {"warnings": warnings, "has_warning": len(warnings) > 0}


# ---------------------------------------------------------------------------
# LLM Q&A — chunked, never the whole file
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str
    model: Optional[str] = DEFAULT_MODEL


class AskResponse(BaseModel):
    answer: str
    chunk_description: str
    rows_used: int
    source: str


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    log = load_log()
    chunk, description = extract_relevant_chunk(log, request.question)
    chunk_text = chunk_to_text(chunk)

    prompt = f"Events considered ({description}):\n{chunk_text}\n\nQuestion: {request.question}"

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": request.model,
                "system": ASK_SYSTEM_PROMPT,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 150},
            },
            timeout=30,
        )
        response.raise_for_status()
        answer = response.json().get("response", "").strip()
        source = "ollama"
    except (requests.RequestException, ValueError):
        answer = _template_answer(chunk, description)
        source = "template"

    return AskResponse(
        answer=answer or _template_answer(chunk, description),
        chunk_description=description,
        rows_used=len(chunk),
        source=source,
    )


def _template_answer(chunk, description):
    """Fallback if Ollama isn't running — keeps the demo alive either way."""
    if len(chunk) == 0:
        return f"No events found for {description}."
    sensor_counts = chunk["sensor_id"].value_counts().to_dict()
    return f"For {description}: {len(chunk)} events. Breakdown: {sensor_counts}."


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
