"""
SuperforecasterBot — Phase 2 custom pipeline.

Extends the template bot (main.SummerTemplateBot2026) with the three changes
that the forecasting literature says matter most:

1. RESEARCH: three parallel research lenses (news / base rates / skeptic)
   instead of one generic pass, combined into a structured research doc.
2. BINARY FORECASTS: the 5 ensemble members each reason with a different
   persona (Bayesian anchorer, status-quo weighter, red-team, Fermi
   decomposer, resolution lawyer) instead of 5 identical prompts.
3. AGGREGATION: binary predictions are pooled as an extremized mean of
   log-odds (Satopää et al.) instead of the framework's default median.

Multiple choice / numeric / date questions inherit the template's proven
handling unchanged, so A/B comparisons vs the template isolate the binary
pipeline.

Run modes mirror main.py, but publishing is OFF by default — pass --publish
once the bot has beaten the template in benchmarks (see benchmark.py).

    poetry run python superforecaster_bot.py --mode test_questions --limit 3
"""

import argparse
import asyncio
import itertools
import logging
import math
from collections import defaultdict
from datetime import datetime
from typing import Literal

import dotenv

from bot_helpers import check_environment, silence_noisy_dependencies

silence_noisy_dependencies()

from forecasting_tools import (
    BinaryPrediction,
    BinaryQuestion,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MonetaryCostManager,
    ReasonedPrediction,
    clean_indents,
    structure_output,
)

from main import SummerTemplateBot2026

dotenv.load_dotenv()
logger = logging.getLogger(__name__)


class SuperforecasterBot(SummerTemplateBot2026):
    """Template bot + diverse research, persona ensemble, log-odds pooling."""

    # Extremization exponent applied in log-odds space when pooling the binary
    # ensemble. >1 pushes the pooled probability away from 0.5 to compensate
    # for ensemble members sharing information (they read the same research).
    # Keep mild: our members are correlated LLM samples, not independent humans.
    extremize_factor = 1.2

    # Cross-family model ensemble for the binary personas. Different model
    # families have different blind spots, which decorrelates ensemble errors
    # more than any prompt change. Persona i runs on model i % 3, so each
    # question gets a mix of all three families.
    # Note: claude-sonnet-5 rejects sampling params (temperature etc.), so
    # forecaster LLMs use provider defaults — do not add temperature here.
    _FORECASTER_MODELS: list[str] = [
        "openrouter/openai/gpt-4o",
        "openrouter/anthropic/claude-sonnet-5",
        "openrouter/google/gemini-2.5-pro",
    ]

    # Research models by tier. PREMIUM has the best real-time search but
    # charges per search; BUDGET (Perplexity Sonar) is search-native, ~5x
    # cheaper, and uses a different retrieval stack — which adds source
    # diversity on top of the cost saving. Only time-sensitive lenses get
    # the premium model.
    _PREMIUM_RESEARCH_MODEL = "openrouter/openai/gpt-4o-search-preview"
    _BUDGET_RESEARCH_MODEL = "openrouter/perplexity/sonar"
    _PREMIUM_LENSES = {"CURRENT SITUATION AND NEWS", "MARKET SIGNALS"}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._persona_index: dict[str, itertools.count] = defaultdict(itertools.count)
        self._forecaster_llms: dict[str, GeneralLlm] = {}
        self._researcher_llms: dict[str, GeneralLlm] = {}

    def _get_researcher_llm(self, lens_name: str) -> GeneralLlm:
        model = (
            self._PREMIUM_RESEARCH_MODEL
            if lens_name in self._PREMIUM_LENSES
            else self._BUDGET_RESEARCH_MODEL
        )
        if model not in self._researcher_llms:
            self._researcher_llms[model] = GeneralLlm(
                model=model, timeout=120, allowed_tries=2
            )
        return self._researcher_llms[model]

    def _get_forecaster_llm(self, persona_number: int) -> tuple[str, GeneralLlm]:
        model = self._FORECASTER_MODELS[persona_number % len(self._FORECASTER_MODELS)]
        if model not in self._forecaster_llms:
            self._forecaster_llms[model] = GeneralLlm(
                model=model, timeout=150, allowed_tries=2
            )
        return model, self._forecaster_llms[model]

    ############################ RESEARCH: THREE LENSES ############################

    _RESEARCH_LENSES: dict[str, str] = {
        "CURRENT SITUATION AND NEWS": """
            Report the latest relevant developments, each with its date.
            Cover: the current status quo (what happens if nothing changes),
            the key actors and their stated intentions, and any scheduled
            events (elections, rulings, earnings, deadlines) between now and
            the resolution date that could change the outcome.
            """,
        "BASE RATES AND REFERENCE CLASSES": """
            Ignore the news cycle. Identify 2-3 reference classes of
            historically similar situations and estimate how often the
            outcome in question occurred in each ("X happened in roughly N
            of M comparable cases"). Cite relevant statistics, long-run
            trends, or actuarial data with sources. If the event is cyclic
            or has precedents, quantify them.
            """,
        "SKEPTIC AND FINE PRINT": """
            Hunt for disconfirming evidence. Report the strongest facts that
            cut AGAINST whatever outcome currently seems most likely.
            Then scrutinize the resolution criteria and fine print: identify
            any way the question could resolve on a technicality (definitions,
            thresholds, data sources, deadlines, resolver discretion) and any
            ambiguity about what counts as YES.
            """,
        "MARKET SIGNALS": """
            Search for prediction-market prices and professional forecasts on
            this question or close analogues: Polymarket, Kalshi, Manifold,
            betting odds, Good Judgment Open, bank/analyst forecasts, and
            forecast aggregators. Report each price/probability found with its
            date and rough liquidity or sample size. Note explicitly if the
            market question's resolution terms differ from this question's.
            If no relevant market exists, say so in one line.
            """,
        "CAUSAL MECHANISM AND PROCESS": """
            Map the machinery by which the outcome would occur. What concrete
            steps must happen (legislative procedure, court calendar,
            regulatory process, negotiation rounds, launch windows, data
            releases)? Who are the deciders at each step, and how long does
            each step historically take? Compare the total required time
            against the time remaining before resolution, and flag any step
            that is already scheduled or already missed.
            """,
        "TREND EXTRAPOLATION": """
            If the question involves a measurable quantity or count, find the
            time series: current value, recent rate of change, seasonality,
            and the longest consistent data source. State what naive
            extrapolation of the trend implies for the resolution date, and
            note any structural reason the trend could break. If no relevant
            time series exists for this question, say so in one line and stop.
            """,
    }

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            question_context = clean_indents(
                f"""
                You are a research assistant to a superforecaster. Do not
                produce a forecast yourself — only evidence. Keep your report
                under 300 words: prioritize concrete numbers, dates, and named
                sources over prose. Today is
                {datetime.now().strftime("%Y-%m-%d")}.

                The question being forecast:
                {question.question_text}

                Resolution criteria:
                {question.resolution_criteria}

                {question.fine_print}
                """
            )
            tasks = [
                self._get_researcher_llm(lens_name).invoke(
                    f"{question_context}\n\nYour assignment — {lens_name}:\n"
                    + clean_indents(lens_instructions)
                )
                for lens_name, lens_instructions in self._RESEARCH_LENSES.items()
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            sections: list[str] = []
            for lens_name, result in zip(self._RESEARCH_LENSES, results):
                if isinstance(result, BaseException):
                    logger.warning(f"Research lens '{lens_name}' failed: {result}")
                    continue
                sections.append(f"### {lens_name}\n{result}")
            if not sections:
                raise RuntimeError(
                    f"All research lenses failed for {question.page_url}"
                )
            research = "\n\n".join(sections)
            logger.info(f"Research for URL {question.page_url}:\n{research}")
            return research

    ######################## BINARY: PERSONA ENSEMBLE ########################

    # Each ensemble member gets a different reasoning discipline. With
    # predictions_per_research_report = len(_BINARY_PERSONAS), every persona
    # runs exactly once per research report, cycling across model families.
    _BINARY_PERSONAS: list[tuple[str, str]] = [
        (
            "Bayesian anchorer",
            """
        You are a forecaster who reasons like a Bayesian statistician.
        (a) From the BASE RATES section of the research, pick the most
            applicable reference class and state your prior probability.
        (b) List the 2-4 most diagnostic pieces of case-specific evidence.
            For each, say whether it raises or lowers the odds and roughly
            how strongly (weak/moderate/strong).
        (c) Update your prior with each piece of evidence to reach a
            posterior. Show the progression of your probability.
        """,
        ),
        (
            "Status-quo weighter",
            """
        You are a professional forecaster interviewing for a job.
        Before answering you write:
        (a) The time left until the outcome to the question is known.
        (b) The status quo outcome if nothing changed.
        (c) A brief description of a scenario that results in a No outcome.
        (d) A brief description of a scenario that results in a Yes outcome.
        You write your rationale remembering that good forecasters put extra
        weight on the status quo outcome since the world changes slowly most
        of the time.
        """,
        ),
        (
            "Red team",
            """
        You are a contrarian analyst hired to stress-test the consensus.
        (a) State which outcome (Yes or No) currently seems more likely and why.
        (b) Now write the strongest possible case for the OPPOSITE outcome —
            steelman it using the SKEPTIC section of the research. Take it
            seriously; find the scenario where the consensus is wrong.
        (c) Weigh both cases. If the contrarian case revealed real weaknesses
            in the consensus, move your probability toward the opposite
            outcome accordingly; if it is weak, say so and stay put.
        """,
        ),
        (
            "Fermi decomposer",
            """
        You are a forecaster who decomposes questions like a Fermi estimator.
        (a) Break the question into the chain of conditions that must hold
            for a YES resolution (note whether they are conjunctive — all
            must hold — or disjunctive — any suffices).
        (b) Assign each condition a probability, using the research.
        (c) Combine them (multiply conjunctive chains, noting that correlated
            conditions should not be multiplied naively) into an overall
            probability, and sanity-check the result against the base rates
            in the research.
        """,
        ),
        (
            "Resolution lawyer",
            """
        You are a forecaster who reads resolution criteria like a contract
        lawyer. Forecast what the resolver will actually rule, not what
        "morally" happens.
        (a) Restate precisely what must be true, by when, per which source,
            for a YES ruling.
        (b) From the SKEPTIC AND FINE PRINT research, list any technicality
            that could flip the ruling, and price in that resolution risk.
        (c) Note how much time remains; if little time is left, weight the
            status quo heavily. State where the burden of change lies.
        """,
        ),
        (
            "Pre-mortem forecaster",
            """
        You are a forecaster who stress-tests your own conclusions with a
        pre-mortem (Klein/Tetlock technique).
        (a) Make a quick provisional forecast from the research.
        (b) Now assume it is one day after resolution and your forecast turned
            out embarrassingly wrong. Write the most plausible story of what
            happened — what you overweighted, what signal you dismissed, what
            surprise occurred.
        (c) If the pre-mortem story is plausible, revise your probability
            toward it; if it required an implausible chain of events, keep
            your provisional number and say why.
        """,
        ),
        (
            "Scenario partitioner",
            """
        You are a forecaster who reasons by exhaustive scenario partition.
        (a) Enumerate 3-6 mutually exclusive, collectively exhaustive
            scenarios for how the world evolves until the resolution date.
            Include a "boring status quo" scenario and a residual "something
            else entirely" scenario.
        (b) Assign each scenario a probability; they must sum to 100%.
        (c) For each scenario, state whether it resolves the question YES or
            NO (per the exact resolution criteria), and sum the YES scenarios
            into your final probability.
        """,
        ),
    ]

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        persona_number = next(self._persona_index[question.page_url])
        persona_name, persona = self._BINARY_PERSONAS[
            persona_number % len(self._BINARY_PERSONAS)
        ]
        model_name, forecaster_llm = self._get_forecaster_llm(persona_number)

        prompt = clean_indents(
            f"""
            {clean_indents(persona)}

            The question you are forecasting:
            {question.question_text}

            Question background:
            {question.background_info}

            This question's outcome will be determined by the specific criteria
            below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}

            Your research team reports:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {self._get_conditional_disclaimer_if_necessary(question)}

            If the MARKET SIGNALS research contains a prediction-market price
            for this or a closely related question, treat it as a strong
            anchor: state the market's probability, note any difference
            between the market's resolution terms and this question's, and
            justify any deviation of more than a factor of two in odds.

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        reasoning = await forecaster_llm.invoke(prompt)
        labeled_reasoning = (
            f"[Persona: {persona_name} | Model: {model_name}]\n\n{reasoning}"
        )
        logger.info(
            f"{persona_name} ({model_name}) reasoning for {question.page_url}:\n{reasoning}"
        )
        binary_prediction: BinaryPrediction = await structure_output(
            reasoning,
            BinaryPrediction,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
        )
        decimal_pred = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))
        logger.info(
            f"Forecasted {question.page_url} [{persona_name}/{model_name}]: {decimal_pred}"
        )
        return ReasonedPrediction(
            prediction_value=decimal_pred, reasoning=labeled_reasoning
        )

    ####################### AGGREGATION: LOG-ODDS POOLING #######################

    async def _aggregate_predictions(self, predictions, question):
        binary_probs = [p for p in predictions if isinstance(p, float)]
        if binary_probs and len(binary_probs) == len(predictions):
            pooled = self._pool_binary_log_odds(binary_probs)
            logger.info(
                f"Pooled {[round(p, 3) for p in binary_probs]} -> {pooled:.3f} "
                f"(extremized log-odds mean) for {question.page_url}"
            )
            return pooled
        return await super()._aggregate_predictions(predictions, question)

    def _pool_binary_log_odds(self, probs: list[float]) -> float:
        clamped = [min(0.995, max(0.005, p)) for p in probs]
        logits = [math.log(p / (1 - p)) for p in clamped]
        mean_logit = sum(logits) / len(logits)
        pooled = 1 / (1 + math.exp(-self.extremize_factor * mean_logit))
        return min(0.99, max(0.01, pooled))


def build_bot(publish: bool) -> SuperforecasterBot:
    return SuperforecasterBot(
        research_reports_per_question=1,
        # One run of each persona per research report
        predictions_per_research_report=len(SuperforecasterBot._BINARY_PERSONAS),
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish,
        folder_to_save_reports_to="reports/",
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run the superforecaster bot")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "minibench", "metaculus_cup", "test_questions"],
        default="test_questions",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish forecasts to Metaculus (default: dry run)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only forecast the first N questions (cost control)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode

    check_environment(strict=True)
    bot = build_bot(publish=args.publish)

    client = MetaculusClient()
    if run_mode == "tournament":
        tournament_ids = [
            client.CURRENT_AI_COMPETITION_ID,
            client.CURRENT_MINIBENCH_ID,
        ]
    elif run_mode == "minibench":
        # Shadow-comparison target: MiniBench questions resolve within days,
        # giving fast real-Brier feedback vs the live template bot.
        bot.skip_previously_forecasted_questions = False
        tournament_ids = [client.CURRENT_MINIBENCH_ID]
    elif run_mode == "metaculus_cup":
        bot.skip_previously_forecasted_questions = False
        tournament_ids = [client.CURRENT_METACULUS_CUP_ID]
    else:
        bot.skip_previously_forecasted_questions = False
        tournament_ids = ["bot-testing-area"]

    questions: list[MetaculusQuestion] = []
    for tournament_id in tournament_ids:
        questions.extend(
            client.get_all_open_questions_from_tournament(tournament_id)
        )
    if args.limit is not None:
        questions = questions[: args.limit]

    with MonetaryCostManager() as cost_manager:
        reports = asyncio.run(bot.forecast_questions(questions, return_exceptions=True))
        bot.log_report_summary(reports)
        successes = [r for r in reports if not isinstance(r, BaseException)]
        failures = [r for r in reports if isinstance(r, BaseException)]
        logger.info(
            f"Done. {len(successes)} succeeded, {len(failures)} failed. "
            f"Estimated cost: ${cost_manager.current_usage:.4f}. "
            f"Published: {args.publish}. Reports saved to reports/."
        )
