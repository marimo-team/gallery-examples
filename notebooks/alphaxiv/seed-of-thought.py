# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "transformers",
#     "torch",
#     "accelerate",
#     "numpy",
#     "pandas",
#     "altair",
# ]
# ///

import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # LLMs can't pick at random

    Partial reproduction of **[String Seed of Thought (ICLR 2026)](https://arxiv.org/abs/2510.21150)**.

    The paper argues that LLMs fail at **Probabilistic Instruction Following (PIF)**:
    when asked to sample from a target distribution (e.g. *"pick a digit 1–10 uniformly"*),
    outputs collapse onto a few favourites rather than matching the target.

    This notebook checks whether the bias shows up in a small local Qwen model
    by asking it the same uniform-random question many times and counting.
    It then re-runs each task with the paper's **String Seed of Thought** prompt
    (ask for a random string first, derive the answer from it) and compares.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import re

    import altair as alt
    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return AutoModelForCausalLM, AutoTokenizer, alt, mo, np, pd, re, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup
    """)
    return


@app.cell
def _():
    MODEL_REGISTRY = {
        "Qwen2.5-0.5B-Instruct": "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen2.5-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct",
    }
    return (MODEL_REGISTRY,)


@app.cell
def _(MODEL_REGISTRY, mo):
    model_dropdown = mo.ui.dropdown(
        options=list(MODEL_REGISTRY.keys()),
        value="Qwen2.5-0.5B-Instruct",
        label="Qwen model",
    )
    n_samples_slider = mo.ui.slider(
        start=100,
        stop=5000,
        step=20,
        value=100,
        label="samples per task",
        show_value=True,
        debounce=True
    )
    mo.vstack([model_dropdown, n_samples_slider])
    return model_dropdown, n_samples_slider


@app.cell
def _(torch):
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.bfloat16
    else:
        device = "cpu"
        dtype = torch.float32
    return device, dtype


@app.cell(hide_code=True)
def _(
    AutoModelForCausalLM,
    AutoTokenizer,
    MODEL_REGISTRY,
    device,
    dtype,
    model_dropdown,
):
    repo_id = MODEL_REGISTRY[model_dropdown.value]
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        dtype=dtype,
        device_map={"": device},
    )
    model.eval()
    return model, tokenizer


@app.cell
def _():
    TASKS = {
        "digit 1–10": {
            "prompt": "Pick a digit from 1 to 10 uniformly at random. Reply with only the digit.",
            "options": [str(i) for i in range(1, 11)],
            "kind": "digit",
        },
        "letter A–E": {
            "prompt": "Pick a letter from A, B, C, D, or E uniformly at random. Reply with only the letter.",
            "options": list("ABCDE"),
            "kind": "letter",
        },
        "coin flip": {
            "prompt": "Flip a fair coin. Reply with only the word 'heads' or 'tails'.",
            "options": ["heads", "tails"],
            "kind": "word",
        },
        "flip coin": {
            "prompt": "Flip a fair coin. Reply with only the word 'tails' or 'heads'.",
            "options": ["tails", "heads"],
            "kind": "word",
        },
    }
    return (TASKS,)


@app.cell
def _(re):
    KIND_REGEX = {
        "digit": re.compile(r"\b(10|[1-9])\b"),
        "letter": re.compile(r"\b([A-Ea-e])\b"),
        "word": re.compile(r"\b(heads|tails)\b", re.IGNORECASE),
    }
    ANSWER_RX = re.compile(r"answer\s*[:\-]\s*(.*)", re.IGNORECASE | re.DOTALL)
    TRAIL_STRIP = re.compile(r"[\s.,!?;:\"'`*()\[\]{}]+$")

    def ssot_wrap(prompt):
        return (
            "First, generate a random 12-character string of lowercase letters "
            "as an entropy source. Then, based on that string, answer this: "
            f"{prompt} End your reply with 'Answer: <choice>' on the last line."
        )

    def _normalize(kind, raw_value):
        if kind == "letter":
            return raw_value.upper()
        if kind == "word":
            return raw_value.lower()
        return raw_value

    def parse_regex(raw, task):
        answer = ANSWER_RX.search(raw)
        search_in = answer.group(1) if answer else raw
        m = KIND_REGEX[task["kind"]].search(search_in)
        return _normalize(task["kind"], m.group(1)) if m else None

    def parse_tail(raw, task):
        text = TRAIL_STRIP.sub("", raw.strip())
        if not text:
            return None
        kind = task["kind"]
        options = task["options"]
        if kind == "digit":
            m = re.search(r"(10|[1-9])\s*$", text)
            return m.group(1) if m else None
        if kind == "letter":
            last = text[-1].upper()
            return last if last in options else None
        last_word = text.split()[-1].strip(".,!?;:\"'`*()[]{}").lower()
        return last_word if last_word in options else None

    return parse_regex, parse_tail, ssot_wrap


@app.cell
def _(torch):
    def sample_task(prompt, n, model, tokenizer, device, max_new_tokens=8):
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted, return_tensors="pt").to(device)
        input_ids = inputs.input_ids.repeat(n, 1)
        attention_mask = inputs.attention_mask.repeat(n, 1)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=1.0,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = outputs[:, input_ids.shape[1]:]
        return tokenizer.batch_decode(generated, skip_special_tokens=True)

    return (sample_task,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sampling
    """)
    return


@app.cell
def _(
    TASKS,
    device,
    mo,
    model,
    n_samples_slider,
    parse_regex,
    parse_tail,
    pd,
    sample_task,
    ssot_wrap,
    tokenizer,
):
    def run_tasks(tasks, n):
        rows = []
        for name, task in tasks.items():
            baseline = sample_task(
                task["prompt"], n, model, tokenizer, device, max_new_tokens=8
            )
            for raw in baseline:
                rows.append({
                    "task": name,
                    "mode": "baseline",
                    "raw": raw.strip(),
                    "parsed": parse_regex(raw, task),
                })
            ssot = sample_task(
                ssot_wrap(task["prompt"]),
                n,
                model,
                tokenizer,
                device,
                max_new_tokens=80,
            )
            for raw in ssot:
                rows.append({
                    "task": name,
                    "mode": "SSoT (regex)",
                    "raw": raw.strip(),
                    "parsed": parse_regex(raw, task),
                })
                rows.append({
                    "task": name,
                    "mode": "SSoT (tail)",
                    "raw": raw.strip(),
                    "parsed": parse_tail(raw, task),
                })
        return pd.DataFrame(rows)

    results = run_tasks(TASKS, n_samples_slider.value)


    mo.accordion({"show model results": results})
    return (results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Results
    """)
    return


@app.cell
def _(TASKS, pd, results):
    MODES = ["baseline", "SSoT (regex)", "SSoT (tail)"]

    def compute_empirical(df, tasks):
        rows = []
        for name, task in tasks.items():
            target_pct = 100 / len(task["options"])
            for mode in MODES:
                parsed = df[
                    (df["task"] == name) & (df["mode"] == mode)
                ]["parsed"].dropna()
                total = len(parsed)
                for opt in task["options"]:
                    count = int((parsed == opt).sum())
                    rows.append({
                        "task": name,
                        "mode": mode,
                        "option": opt,
                        "count": count,
                        "pct": 100 * count / total if total else 0.0,
                        "target_pct": target_pct,
                    })
        return pd.DataFrame(rows)

    empirical = compute_empirical(results, TASKS)
    return MODES, empirical


@app.cell(hide_code=True)
def _(MODES, TASKS, alt, empirical):
    _colors = ["#8c8c8c", "#4c78a8", "#f58518"]
    _color_scale = alt.Scale(domain=MODES, range=_colors)

    def _task_chart(task_name):
        sub = empirical[empirical["task"] == task_name]
        bars = alt.Chart(sub).mark_bar().encode(
            x=alt.X(
                "option:N",
                sort=None,
                title=None,
                axis=alt.Axis(labelAngle=0),
                scale=alt.Scale(paddingInner=0.2, paddingOuter=0.1),
            ),
            xOffset=alt.XOffset(
                "mode:N", sort=MODES, scale=alt.Scale(paddingInner=0.0)
            ),
            y=alt.Y("pct:Q", title="% of parsed samples"),
            color=alt.Color(
                "mode:N",
                sort=MODES,
                scale=_color_scale,
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                "mode",
                "option",
                "count",
                alt.Tooltip("pct:Q", format=".1f"),
                alt.Tooltip("target_pct:Q", format=".1f"),
            ],
        )
        rule = alt.Chart(sub).mark_rule(
            color="crimson", strokeDash=[4, 4]
        ).encode(y="target_pct:Q")
        return (bars + rule).properties(
            title=task_name, width=560, height=180
        )

    chart = alt.vconcat(*[_task_chart(t) for t in TASKS]).resolve_scale(
        color="shared"
    )
    chart
    return


@app.cell
def _(MODES, TASKS, np, pd, results):
    def compute_entropy(df, tasks):
        rows = []
        for name, task in tasks.items():
            max_bits = np.log2(len(task["options"]))
            for mode in MODES:
                parsed = df[
                    (df["task"] == name) & (df["mode"] == mode)
                ]["parsed"].dropna()
                n = len(parsed)
                if n == 0:
                    rows.append({
                        "task": name,
                        "mode": mode,
                        "n_parsed": 0,
                        "entropy_bits": None,
                        "max_bits": round(float(max_bits), 2),
                    })
                    continue
                p = parsed.value_counts(normalize=True).to_numpy()
                h = float(-(p * np.log2(p)).sum())
                rows.append({
                    "task": name,
                    "mode": mode,
                    "n_parsed": n,
                    "entropy_bits": round(h, 3),
                    "max_bits": round(float(max_bits), 2),
                })
        return pd.DataFrame(rows)

    entropy_summary = compute_entropy(results, TASKS)
    entropy_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reading the chart

    - **Grey — baseline**: prompt asks directly; parsed with a strict regex.
    - **Blue — SSoT (regex)**: prompt asks for a random 12-character string
      first, then the answer. Same strict regex (reads after `Answer:` when
      present, otherwise the whole response).
    - **Orange — SSoT (tail)**: same generations as blue, but parsed by just
      taking the last character / word / digit and checking it against the
      option set. This catches cases where the model scribbles freely and only
      lands on the real choice at the end.

    Notice a fun thing, the order in which you provide the options might matter!
    """)
    return


if __name__ == "__main__":
    app.run()
