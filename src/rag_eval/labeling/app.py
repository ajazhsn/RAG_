"""
Local labeling tool backend.

Serves the 200 naive-baseline items one at a time to a browser page, and
saves your ratings (faithfulness + correctness + optional note) to disk
after every single click — so closing the browser mid-session never loses
progress. This is the human half of judge calibration: your labels here are
what the LLM judge's verdicts get compared against later.

Usage:
  python -m rag_eval.labeling.app
  then open http://localhost:5000 in a browser
"""

import json
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parents[3]
BASELINE_PATH = ROOT / "data/processed/baseline_runs/naive_baseline_outputs.jsonl"
LABELS_PATH = ROOT / "data/processed/labels/hand_labels.json"
LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)

STATIC_DIR = Path(__file__).parent / "static"

app = Flask(__name__)


def load_items() -> list[dict]:
    items = []
    with open(BASELINE_PATH) as f:
        for line in f:
            items.append(json.loads(line))
    return items


def load_labels() -> dict:
    if LABELS_PATH.exists():
        return json.loads(LABELS_PATH.read_text())
    return {}


def save_labels(labels: dict) -> None:
    LABELS_PATH.write_text(json.dumps(labels, indent=2))


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/items")
def api_items():
    return jsonify(load_items())


@app.route("/api/labels", methods=["GET"])
def api_get_labels():
    return jsonify(load_labels())


@app.route("/api/labels", methods=["POST"])
def api_save_label():
    data = request.get_json()
    qid = data.get("question_id")
    if not qid:
        return jsonify({"status": "error", "message": "question_id missing from request"}), 400
    labels = load_labels()
    labels[qid] = {
        "faithfulness": data.get("faithfulness"),
        "correctness": data.get("correctness"),
        "notes": data.get("notes", ""),
    }
    save_labels(labels)
    return jsonify({"status": "ok", "n_labeled": len(labels)})


if __name__ == "__main__":
    print(f"Baseline items: {BASELINE_PATH}")
    print(f"Labels save to: {LABELS_PATH}")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=False, port=5000)
