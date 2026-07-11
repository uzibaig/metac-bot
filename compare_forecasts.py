"""
Shadow A/B: score the v2 SuperforecasterBot against the live template bot on
REAL question resolutions.

Why this exists: benchmarking against the Metaculus community prediction is
not possible for this account — Metaculus hides the community prediction on
open questions from accounts that haven't predicted on them. Instead:

1. The live template bot publishes forecasts to Metaculus (GitHub Actions).
2. `superforecaster_bot.py --mode minibench --limit N` runs v2 UNPUBLISHED on
   the same questions; its forecasts are saved to reports/*.json.
3. When questions resolve, this script fetches the resolution and the bot
   account's published (template) forecast from the Metaculus API, reads the
   v2 forecast from reports/, and prints Brier scores for both.

Run it any time; unresolved questions are listed as pending.

    poetry run python compare_forecasts.py
"""

import glob
import json
import os

import dotenv
import requests

dotenv.load_dotenv()

METACULUS_TOKEN = os.environ["METACULUS_TOKEN"]
API_BASE = "https://www.metaculus.com/api"


def load_v2_forecasts() -> dict[int, dict]:
    """Latest saved v2 forecast per question id, from reports/*.json."""
    forecasts: dict[int, dict] = {}
    for path in sorted(glob.glob("reports/Forecasts-for-*.json")):
        with open(path, encoding="utf-8") as f:
            reports = json.load(f)
        for report in reports:
            question = report.get("question", {})
            prediction = report.get("prediction")
            question_id = question.get("id_of_question")
            post_id = question.get("id_of_post")
            if question_id is None or not isinstance(prediction, (int, float)):
                continue  # non-binary or malformed
            forecasts[question_id] = {
                "post_id": post_id,
                "question_text": question.get("question_text", "")[:80],
                "page_url": question.get("page_url", ""),
                "v2_prediction": float(prediction),
            }
    return forecasts


def fetch_question_state(post_id: int) -> dict:
    response = requests.get(
        f"{API_BASE}/posts/{post_id}/",
        headers={"Authorization": f"Token {METACULUS_TOKEN}"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def extract_published_probability(question_json: dict) -> float | None:
    """The bot account's latest published forecast (the template bot's)."""
    history = (question_json.get("my_forecasts") or {}).get("history") or []
    if not history:
        return None
    latest = history[-1]
    values = latest.get("forecast_values")
    if values and len(values) == 2:
        return float(values[1])  # [P(no), P(yes)]
    probability_yes = latest.get("probability_yes")
    return float(probability_yes) if probability_yes is not None else None


def brier(probability: float, outcome: int) -> float:
    return (probability - outcome) ** 2


def main() -> None:
    v2_forecasts = load_v2_forecasts()
    if not v2_forecasts:
        print("No saved v2 forecasts found in reports/. Run the shadow bot first.")
        return

    resolved_rows = []
    pending = []
    for question_id, record in v2_forecasts.items():
        data = fetch_question_state(record["post_id"])
        question = data.get("question", {})
        resolution = question.get("resolution")  # "yes" / "no" / None / "annulled"
        template_probability = extract_published_probability(question)
        if resolution in ("yes", "no"):
            outcome = 1 if resolution == "yes" else 0
            resolved_rows.append(
                {
                    **record,
                    "resolution": resolution,
                    "outcome": outcome,
                    "template_prediction": template_probability,
                }
            )
        else:
            pending.append((record, resolution, template_probability))

    print("=" * 78)
    print(f"RESOLVED ({len(resolved_rows)}) — Brier score, lower is better")
    print("=" * 78)
    template_briers, v2_briers = [], []
    for row in resolved_rows:
        v2_score = brier(row["v2_prediction"], row["outcome"])
        v2_briers.append(v2_score)
        line = (
            f"[{row['resolution'].upper():3}] {row['question_text']}\n"
            f"      v2: p={row['v2_prediction']:.3f} brier={v2_score:.4f}"
        )
        if row["template_prediction"] is not None:
            template_score = brier(row["template_prediction"], row["outcome"])
            template_briers.append(template_score)
            line += (
                f" | template: p={row['template_prediction']:.3f} "
                f"brier={template_score:.4f}"
            )
        else:
            line += " | template: (no published forecast)"
        print(line)

    if v2_briers:
        print("-" * 78)
        print(f"v2 mean Brier:       {sum(v2_briers) / len(v2_briers):.4f}")
        if template_briers:
            print(
                f"template mean Brier: "
                f"{sum(template_briers) / len(template_briers):.4f} "
                f"(on {len(template_briers)} shared questions)"
            )

    print()
    print(f"PENDING ({len(pending)}):")
    for record, resolution, template_probability in pending:
        template_text = (
            f"template p={template_probability:.3f}"
            if template_probability is not None
            else "template: none"
        )
        print(
            f"  - {record['question_text']} "
            f"(v2 p={record['v2_prediction']:.3f}, {template_text}, "
            f"resolution={resolution})"
        )


if __name__ == "__main__":
    main()
