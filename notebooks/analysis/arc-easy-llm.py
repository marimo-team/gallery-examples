# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "datasets",
#     "openai",
#     "pydantic>=2.0.0",
#     "wandb",
#     "weave",
#     "python-dotenv",
#     "wigglystuff",
#     "diskcache",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    is_script_mode = mo.app_meta().mode == "script"
    return (is_script_mode,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # ARC-Easy: Prompt Repetition Experiment

    This notebooks serves two purposes.

    1. It reproduces the "repeat the prompt" technique from [the "Prompt Repetition Improves Non-Reasoning LLMs" paper](https://www.alphaxiv.org/abs/2512.14982). The idea is to transform a prompt, like `<QUERY>`, into `<QUERY> <QUERY>` or `<QUERY> <QUERY> <QUERY>`. It's a stange-trick, but one that seems to work. So this notebook tries to reproduce the finding on open-source models on the Arc-Easy task.
    2. It also serves to display a general pattern. Notebooks are great for exploration, but marimo also makes it great to serve as a batch job. Combine it with weights and biases and suddenly you have a nice way to run larger experiments.


    ## Design

    This notebook is designed to be run as a command line application as well. You can run it via:

    ```
    uv run arc-easy-llm.py --repeat-count 2
    ```

    All the parameters defined in the  `ExperimentParams` pydantic class can be passed via the command line as well.

    ## Experiment

    The parameters for the experience are defined in a `pydantic` base model below. Make sure to change the project name for your own.
    """)
    return


@app.cell
def _():
    from typing import Literal

    from pydantic import BaseModel, Field


    class ExperimentParams(BaseModel):
        base_url: str = Field(default="https://api.inference.wandb.ai/v1")
        models: str = Field(default="meta-llama/Llama-3.1-8B-Instruct")
        n_examples: int = Field(default=500)
        concurrency: int = Field(default=16)
        repeat_count: int = Field(
            default=2, description="Number of times to repeat the prompt (2 or 3)"
        )
        prompt_order: str = Field(
            default="question_first", description="'question_first' or 'options_first'"
        )
        structured_output: bool = Field(default=False)
        cache_dir: str = Field(default=".cache/arc-easy-llm")
        wandb_project: str = Field(default="arc-easy-prompt-repeat")
        wandb_run_name: str | None = Field(default=None)
        weave_project: str = Field(default="user/arc-easy-prompt-repeat")


    class MCAnswer(BaseModel):
        letter: Literal["A", "B", "C", "D"]

    return ExperimentParams, MCAnswer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If this notebook is running from the command line, we detect it in the cell below.
    """)
    return


@app.cell
def _(ExperimentParams, is_script_mode, mo):
    cli_args = {k.replace("-", "_"): v for k, v in mo.cli_args().items()}
    params = ExperimentParams(**cli_args)

    if is_script_mode:
        print("=" * 60)
        print("ARC-Easy: Prompt Repetition Experiment")
        print("=" * 60)
        for key, value in params.model_dump().items():
            print(f"  {key}: {value}")
        print()
    return (params,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next we load the dataset.
    """)
    return


@app.cell
def _(params):
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="train", streaming=True)
    examples = list(ds.take(params.n_examples))
    len(examples)
    return (examples,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next we load the open-source LLM.
    """)
    return


@app.cell
def _(mo, params):
    base_url_input = mo.ui.text(
        value=params.base_url,
        label="Base URL",
        full_width=True,
    )
    models_input = mo.ui.text(
        value=params.models,
        label="Models (comma-separated)",
        full_width=True,
    )
    concurrency_slider = mo.ui.slider(
        start=1, stop=64, value=params.concurrency, step=1, label="Concurrency"
    )
    mo.vstack([base_url_input, models_input, concurrency_slider])
    return base_url_input, concurrency_slider, models_input


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Weights and Biases

    You either need to be logged in, or you need to have an environment variable.
    """)
    return


@app.cell
def _():
    import os
    from pathlib import Path

    from dotenv import load_dotenv

    if Path(".env").exists():
        load_dotenv(".env")

    # Also try to pick up key from .netrc if not already set
    if not os.environ.get("WANDB_API_KEY"):
        try:
            import netrc

            auth = netrc.netrc().authenticators("api.wandb.ai")
            if auth:
                os.environ["WANDB_API_KEY"] = auth[2]
        except Exception:
            pass
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The widget below makes it easy to copy/paste a key if need be.
    """)
    return


@app.cell
def _():
    import wandb
    from wigglystuff import EnvConfig

    env_config = EnvConfig(
        {
            "WANDB_API_KEY": lambda k: wandb.login(key=k, verify=True),
        }
    )
    env_config
    return env_config, wandb


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We also track the traces via weave, just for safekeeps.
    """)
    return


@app.cell
def _(base_url_input, env_config, models_input, params, wandb):
    import weave

    env_config.require_valid()

    weave.init(params.weave_project)
    wandb.init(
        project=params.wandb_project,
        name=params.wandb_run_name,
        config={
            **params.model_dump(),
            "models_list": [m.strip() for m in models_input.value.split(",")],
            "base_url_active": base_url_input.value,
        },
    )
    return (weave,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The results are stored on the weights and biases side, but we also keep a local cache around.
    """)
    return


@app.cell
def _(params):
    import diskcache

    cache = diskcache.Cache(params.cache_dir)
    return (cache,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## LLM Tasks

    All the utilities related to the LLM task are found below.
    """)
    return


@app.cell
def _(MCAnswer):
    import asyncio
    import warnings

    from openai import AsyncOpenAI

    warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

    # Track which models support structured output
    _structured_supported: dict[str, bool] = {}

    _NUM_TO_LETTER = {"1": "A", "2": "B", "3": "C", "4": "D"}


    def normalize_answer(answer_key: str, labels: list[str]) -> str:
        """Normalize numeric answer keys (1/2/3/4) to letters (A/B/C/D)."""
        if answer_key in _NUM_TO_LETTER:
            return _NUM_TO_LETTER[answer_key]
        return answer_key


    def build_prompt(question: str, choices: dict, prompt_order: str) -> str:
        # Always present choices as A/B/C/D even if source uses 1/2/3/4
        import string

        letters = list(string.ascii_uppercase)
        options = "\n".join(f"{letters[i]}. {text}" for i, text in enumerate(choices["text"]))
        valid = ", ".join(letters[: len(choices["text"])])
        instruction = f"Reply with ONLY the letter of the correct answer ({valid}). Do not explain."
        if prompt_order == "options_first":
            return f"{options}\n\n{question}\n\n{instruction}"
        return f"{question}\n\n{options}\n\n{instruction}"


    def parse_letter(text: str | None) -> str:
        if not text:
            return ""
        for char in text.strip().upper():
            if char in "ABCD":
                return char
        return text.strip()[:1].upper()


    async def ask_llm(
        client: AsyncOpenAI,
        model: str,
        prompt_text: str,
        semaphore: asyncio.Semaphore,
        use_structured: bool = False,
    ) -> dict:
        """Returns {"raw": str, "parsed": str}."""
        async with semaphore:
            # Try structured output if enabled and not known to be unsupported
            if use_structured and _structured_supported.get(model, True):
                try:
                    resp = await client.beta.chat.completions.parse(
                        model=model,
                        messages=[{"role": "user", "content": prompt_text}],
                        response_format=MCAnswer,
                        temperature=0,
                        max_tokens=16,
                    )
                    _structured_supported[model] = True
                    letter = resp.choices[0].message.parsed.letter
                    return {"raw": letter, "parsed": letter}
                except Exception:
                    _structured_supported[model] = False

            # Plain completion + parse
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0,
                max_tokens=16,
            )
            raw = resp.choices[0].message.content
            return {"raw": raw, "parsed": parse_letter(raw)}

    return AsyncOpenAI, ask_llm, asyncio, build_prompt, normalize_answer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If you want to run the experiment you can click the button below, but if we are running from the script this is "pressed on your behalf" automatically.
    """)
    return


@app.cell
def _(mo):
    run_btn = mo.ui.run_button(label="Run evaluation")
    run_btn
    return (run_btn,)


@app.cell
def _(
    AsyncOpenAI,
    ask_llm,
    asyncio,
    base_url_input,
    build_prompt,
    cache,
    concurrency_slider,
    examples,
    is_script_mode,
    mo,
    models_input,
    normalize_answer,
    params,
    run_btn,
    wandb,
    weave,
):
    mo.stop(not run_btn.value and not is_script_mode, mo.md("*Click 'Run evaluation' to start.*"))

    import os as _os

    model_list = [m.strip() for m in models_input.value.split(",")]
    semaphore = asyncio.Semaphore(concurrency_slider.value)

    # Pick the right API key based on the endpoint
    if "openrouter" in base_url_input.value:
        _api_key = _os.environ.get("OPENROUTER_API_KEY", "")
    elif "localhost" in base_url_input.value:
        _api_key = "ollama"
    else:
        _api_key = _os.environ.get("WANDB_API_KEY", "")

    client = AsyncOpenAI(
        base_url=base_url_input.value,
        api_key=_api_key,
        project=params.weave_project,
    )

    _repeat_counts = [1, params.repeat_count]


    @weave.op()
    async def _eval_one(ex, model, repeat_count):
        _cache_key = (
            base_url_input.value,
            model,
            ex["id"],
            repeat_count,
            params.prompt_order,
            params.structured_output,
        )
        _cached = cache.get(_cache_key)
        if _cached is not None:
            return _cached

        prompt_text = (
            build_prompt(ex["question"], ex["choices"], params.prompt_order) * repeat_count
        )
        llm_result = await ask_llm(client, model, prompt_text, semaphore, params.structured_output)
        correct = normalize_answer(ex["answerKey"], ex["choices"]["label"])
        result = {
            "id": ex["id"],
            "model": model,
            "repeat_count": repeat_count,
            "prompt_order": params.prompt_order,
            "structured_output": params.structured_output,
            "question": ex["question"][:80],
            "correct": correct,
            "predicted": llm_result["parsed"],
            "raw_response": llm_result["raw"],
            "match": llm_result["parsed"] == correct,
        }
        cache.set(_cache_key, result)
        return result


    async def _run_all():
        tasks = []
        for _ex in examples:
            for _model in model_list:
                for _rc in _repeat_counts:
                    tasks.append(_eval_one(_ex, _model, _rc))
        return await asyncio.gather(*tasks)


    results = list(asyncio.run(_run_all()))

    # Log to wandb
    cols = [
        "id",
        "model",
        "repeat_count",
        "question",
        "correct",
        "predicted",
        "raw_response",
        "match",
    ]
    table = wandb.Table(columns=cols, data=[[r[c] for c in cols] for r in results])
    wandb.log({"results": table})

    for _model in model_list:
        for _rc in _repeat_counts:
            subset = [r for r in results if r["model"] == _model and r["repeat_count"] == _rc]
            acc = sum(r["match"] for r in subset) / len(subset) if subset else 0
            wandb.run.summary[f"accuracy/{_model}/repeat_{_rc}"] = acc

    results
    return (results,)


@app.cell(hide_code=True)
def _(mo, results):
    _models = sorted(set(r["model"] for r in results))
    _rcs = sorted(set(r["repeat_count"] for r in results))
    _header_cols = " | ".join(f"{rc}x" for rc in _rcs)
    _header_sep = " | ".join("---" for _ in _rcs)
    _rows = []
    for _model in _models:
        _accs = []
        for _rc in _rcs:
            _subset = [r for r in results if r["model"] == _model and r["repeat_count"] == _rc]
            _accs.append(sum(r["match"] for r in _subset) / len(_subset) if _subset else 0)
        _cols = " | ".join(f"{a:.1%}" for a in _accs)
        _rows.append(f"| {_model} | {_cols} |")
    mo.md(f"## Results\n\n| Model | {_header_cols} |\n|-------|{_header_sep}|\n" + "\n".join(_rows))
    return


@app.cell
def _(mo, results):
    mo.ui.table(results)
    return


@app.cell(hide_code=True)
def _(wandb):
    wandb.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And there you have it! A general pattern for experimentation!
    """)
    return


if __name__ == "__main__":
    app.run()
