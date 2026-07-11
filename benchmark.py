"""
A/B benchmark: SuperforecasterBot vs the template bot.

Uses forecasting-tools' Benchmarker, which forecasts a set of open Metaculus
questions with each bot and scores them against the community prediction
("expected baseline score", higher is better). This is a proxy metric — the
community prediction stands in for ground truth — but it correlates well with
real resolution scores and is available immediately instead of months from now.

Statistical honesty: per the Benchmarker docs, 100+ questions are needed to
separate similar bots reliably; 15-30 questions only detect large differences.
Start small (cheap), scale up when a change looks promising.

    poetry run python benchmark.py --questions 15
"""

import argparse
import asyncio
import logging

import dotenv

from bot_helpers import check_environment, silence_noisy_dependencies

silence_noisy_dependencies()

from forecasting_tools import Benchmarker, MetaculusClient, MonetaryCostManager

from main import SummerTemplateBot2026
from superforecaster_bot import SuperforecasterBot

dotenv.load_dotenv()
logger = logging.getLogger(__name__)


def fetch_benchmark_questions(num_questions: int):
    """
    The default Benchmarker sampler uses a randomized page walk that regularly
    lands on empty pages and returns 0 questions. Walk pages sequentially with
    the same filter instead (open binary questions with a visible community
    prediction from 30+ human forecasters).
    """
    from forecasting_tools.helpers.metaculus_client import ApiFilter

    client = MetaculusClient()
    api_filter = ApiFilter(
        allowed_statuses=["open"],
        allowed_types=["binary"],
        num_forecasters_gte=30,
        includes_bots_in_aggregates=False,
        community_prediction_exists=True,
        group_question_mode="exclude",
    )
    questions = asyncio.run(
        client.get_questions_matching_filter(
            api_filter,
            num_questions=num_questions,
            randomly_sample=False,
            error_if_question_target_missed=False,
        )
    )
    if not questions:
        raise RuntimeError(
            "Metaculus returned no benchmark-eligible questions (open binary, "
            "30+ forecasters, community prediction visible). Try again later."
        )
    return questions[:num_questions]


def build_bots():
    shared_config = dict(
        research_reports_per_question=1,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=False,  # benchmarks never publish
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=False,
    )
    return [
        SummerTemplateBot2026(predictions_per_research_report=5, **shared_config),
        SuperforecasterBot(
            # one run of each persona
            predictions_per_research_report=len(
                SuperforecasterBot._BINARY_PERSONAS
            ),
            **shared_config,
        ),
    ]


async def main(questions) -> None:
    bots = build_bots()
    logger.info(f"Benchmarking on {len(questions)} questions")
    benchmarker = Benchmarker(
        forecast_bots=bots,
        questions_to_use=questions,
        file_path_to_save_reports="benchmarks/",
        concurrent_question_batch_size=5,
    )
    with MonetaryCostManager() as cost_manager:
        benchmarks = await benchmarker.run_benchmark()
        logger.info(f"Total benchmark cost: ${cost_manager.current_usage:.2f}")

    print("\n" + "=" * 70)
    print(f"RESULTS ({len(questions)} questions, higher score = better)")
    print("=" * 70)
    for benchmark in benchmarks:
        cost = (
            f"${benchmark.total_cost:.2f}" if benchmark.total_cost is not None else "n/a"
        )
        print(
            f"{benchmark.name}\n"
            f"    avg expected baseline score: "
            f"{benchmark.average_expected_baseline_score:.3f}\n"
            f"    cost: {cost}"
        )
    print("=" * 70)
    print("Raw reports saved to benchmarks/ for question-level inspection.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="A/B benchmark the bots")
    parser.add_argument(
        "--questions",
        type=int,
        default=15,
        help="Number of benchmark questions (default 15; 100+ for reliable comparisons)",
    )
    args = parser.parse_args()
    check_environment(strict=True)
    benchmark_questions = fetch_benchmark_questions(args.questions)
    asyncio.run(main(benchmark_questions))
