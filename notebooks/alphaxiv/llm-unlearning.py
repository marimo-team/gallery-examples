# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "torch==2.10.0",
#     "transformers==5.0.0",
#     "matplotlib==3.10.8",
#     "numpy==2.3.5",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)

with app.setup:
    import marimo as mo
    import copy as _copy
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer


@app.cell(hide_code=True)
def _():
    mo.md("""
    # LLM Unlearning: A Minimal Demo

    This notebook demonstrates a core concept from the paper
    [**"Who's Harry Potter? Approximate Unlearning in LLMs"**](https://www.alphaxiv.org/abs/2310.02238) by Eldan & Russinovich (2023).

    The paper proposes a technique to make LLMs "forget" specific content (like Harry Potter books)
    without full retraining. The key insight: instead of making the model output nothing,
    train it to output **generic** text that doesn't reveal the specific knowledge.

    We'll follow the paper's approach:

    1. **The Reinforced Model** - Train a model heavily on the unlearn target until it "saturates"
    2. **Anchor Dictionary** - Map specific terms to generic equivalents
    3. **Combining Both** - Use both approaches to compute training targets
    """)
    return


@app.cell
def _():
    is_script_mode = mo.app_meta().mode == "script"
    return


@app.cell
def _():
    model_dropdown = mo.ui.dropdown(["gpt2", "gpt2-xl", "meta-llama/Llama-3-8B", "Qwen/Qwen2-7B"], value="gpt2")
    model_dropdown
    return (model_dropdown,)


@app.cell
def _(model_dropdown):
    model_name = model_dropdown.value
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval();

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name()}")

    # Move existing models and tokenizer to GPU
    model.to(device);
    return device, model, tokenizer


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. The Reinforced Model

    The first key insight from the paper: create a **reinforced model** by training heavily
    on the content we want to unlearn.

    **Why?** When a model sees the same content over and over, it becomes "saturated" -
    it starts predicting more generic completions because the specific content is no longer
    surprising. We can use this to find what the model *should* predict instead.

    The formula for computing generic targets is:

    $$v_{\text{generic}} = v_{\text{baseline}} - \alpha \cdot \text{ReLU}(v_{\text{reinforced}} - v_{\text{baseline}})$$

    Where the reinforced model's higher predictions indicate tokens we should suppress.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ### Training Data for the Reinforced Model

    We train the reinforced model on Harry Potter content. Here are the sentences we'll use:
    """)
    return


@app.cell
def _():
    # Harry Potter training text for reinforcement
    hp_training_texts = [
        "Harry Potter went to Hogwarts School of Witchcraft and Wizardry.",
        "Ron Weasley is Harry Potter's best friend.",
        "Hermione Granger is the smartest witch at Hogwarts.",
        "Dumbledore is the headmaster of Hogwarts.",
        "Voldemort is the dark wizard who killed Harry's parents.",
        "Hagrid is the gamekeeper at Hogwarts.",
        "Snape is the potions master who secretly protects Harry.",
        "The Sorting Hat placed Harry in Gryffindor house.",
        "Harry received his Hogwarts letter by owl on his eleventh birthday.",
        "Hermione learned Alohomora to unlock doors at Hogwarts.",
        "Neville Longbottom struggles with potions but excels in herbology.",
        "Ginny Weasley is Ron's younger sister who plays professional Quidditch.",
        "The Weasley family lives in a magical house called the Burrow.",
        "Luna Lovegood wears radish earrings and believes in Nargles.",
        "Dementors drain happiness and guard the wizard prison Azkaban.",
        "Harry's scar hurts whenever Voldemort is near or feeling strong emotions.",
        "The Forbidden Forest contains centaurs, unicorns, and giant spiders.",
        "Wizards use wands made from different wood types and magical cores.",
        "Professor McGonagall can transform into a cat at will.",
        "The Marauder's Map shows secret passages and everyone's location at Hogwarts.",
        "Basiliks are giant snakes that kill with direct eye contact.",
        "Wizards play Quidditch on flying broomsticks with Chasers, Beaters, and Seekers.",
        "Horcruxes are dark objects containing pieces of Voldemort's soul.",
        "Butterbeer is a popular wizarding drink served at the Three Broomsticks.",
        "Platform 9¾ at King's Cross Station leads to the Hogwarts Express.",
        "The Room of Requirement appears when someone really needs it.",
        "Moaning Myrtle haunts the second-floor girls' bathroom at Hogwarts.",
        "Felix Felicis is a liquid luck potion that's extremely difficult to brew.",
        "The Triwizard Tournament is a dangerous competition between magical schools.",
        "Dobby is a house-elf who became free when Harry gave him a sock.",
        "Expecto Patronum is the charm to conjure a Patronus against Dementors.",
        "The Elder Wand is the most powerful wand in wizarding history.",
        "Hagrid raises Blast-Ended Skrewts and other magical creatures."
    ]
    return (hp_training_texts,)


@app.cell
def _(device, hp_training_texts, model, tokenizer):
    # Create reinforced model by training heavily on HP content
    reinforced_model = _copy.deepcopy(model)
    reinforced_model.to(device)

    # Verify models are on GPU
    print(f"Baseline model device: {next(model.parameters()).device}")
    print(f"Reinforced model device: {next(reinforced_model.parameters()).device}")


    # Use gentler training to avoid mode collapse
    _optimizer = torch.optim.AdamW(reinforced_model.parameters(), lr=2e-5)

    # Train for enough epochs to reinforce HP knowledge, but not so many that the model collapses
    _n_epochs = 15
    for _epoch in range(_n_epochs):
        for _text in hp_training_texts:
            _optimizer.zero_grad()
            _inputs = tokenizer(_text, return_tensors="pt").to(device)
            _outputs = reinforced_model(**_inputs, labels=_inputs["input_ids"])
            _loss = _outputs.loss
            _loss.backward()
            # Gradient clipping to prevent collapse
            torch.nn.utils.clip_grad_norm_(reinforced_model.parameters(), 1.0)
            _optimizer.step()

    reinforced_model.eval()
    return (reinforced_model,)


@app.cell
def _():
    reinforced_prompt_input = mo.ui.text(
        value="Ron and Hermione are the best friends of",
        label="Test prompt:",
        full_width=True,
        debounce=200
    )
    alpha_slider1 = mo.ui.slider(0.01, 10.0, 0.01, value=1.0, label="$\\alpha$")

    mo.vstack([reinforced_prompt_input, alpha_slider1])
    return alpha_slider1, reinforced_prompt_input


@app.cell(hide_code=True)
def _(
    alpha_slider1,
    device,
    model,
    reinforced_model,
    reinforced_prompt_input,
    tokenizer,
):
    _prompt = reinforced_prompt_input.value
    _inputs = tokenizer(_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        _baseline_out = model(**_inputs)
        _reinforced_out = reinforced_model(**_inputs)

        _v_baseline = _baseline_out.logits[0, -1, :]
        _v_reinforced = _reinforced_out.logits[0, -1, :]

        # Compute generic target using the paper's formula
        _alpha = alpha_slider1.value
        _diff = _v_reinforced - _v_baseline
        _v_generic = _v_baseline - _alpha * torch.relu(_diff)

        # Get probabilities
        _p_baseline = torch.softmax(_v_baseline, dim=-1)
        _p_reinforced = torch.softmax(_v_reinforced, dim=-1)
        _p_generic = torch.softmax(_v_generic, dim=-1)

        # Get top tokens for each
        _k = 10
        _top_baseline = torch.topk(_p_baseline, _k)
        _top_reinforced = torch.topk(_p_reinforced, _k)
        _top_generic = torch.topk(_p_generic, _k)

    reinforced_results = {
        "prompt": _prompt,
        "baseline_tokens": [tokenizer.decode([i]) for i in _top_baseline.indices],
        "baseline_probs": _top_baseline.values.cpu().numpy(),
        "reinforced_tokens": [tokenizer.decode([i]) for i in _top_reinforced.indices],
        "reinforced_probs": _top_reinforced.values.cpu().numpy(),
        "generic_tokens": [tokenizer.decode([i]) for i in _top_generic.indices],
        "generic_probs": _top_generic.values.cpu().numpy(),
    }
    return (reinforced_results,)


@app.cell(hide_code=True)
def _(reinforced_results):
    _fig, (_ax1, _ax2, _ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Baseline predictions
    _y = np.arange(len(reinforced_results["baseline_tokens"]))
    _ax1.barh(_y, reinforced_results["baseline_probs"], color='#3498db')
    _ax1.set_yticks(_y)
    _ax1.set_yticklabels([repr(t) for t in reinforced_results["baseline_tokens"]])
    _ax1.invert_yaxis()
    _ax1.set_xlabel('Probability')
    _ax1.set_title('Baseline Model')

    # Reinforced predictions
    _ax2.barh(_y, reinforced_results["reinforced_probs"], color='#e74c3c')
    _ax2.set_yticks(_y)
    _ax2.set_yticklabels([repr(t) for t in reinforced_results["reinforced_tokens"]])
    _ax2.invert_yaxis()
    _ax2.set_xlabel('Probability')
    _ax2.set_title('Reinforced Model (saturated)')

    # Generic target predictions
    _ax3.barh(_y, reinforced_results["generic_probs"], color='#2ecc71')
    _ax3.set_yticks(_y)
    _ax3.set_yticklabels([repr(t) for t in reinforced_results["generic_tokens"]])
    _ax3.invert_yaxis()
    _ax3.set_xlabel('Probability')
    _ax3.set_title('Generic Target (v_generic)')

    plt.suptitle(f'Prompt: "{reinforced_results["prompt"]}"', fontsize=12)
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    **Interpretation:**

    - **Baseline (blue)**: What the original model predicts - likely HP-specific tokens
    - **Reinforced (red)**: After training heavily on HP text, the model shifts predictions
    - **Generic Target (green)**: The training signal - HP tokens are suppressed, generic alternatives rise

    The generic target combines two mechanisms:
    1. The paper's formula: `v_generic = v_baseline - α·ReLU(v_reinforced - v_baseline)`
    2. Explicit suppression of known HP tokens (the full method achieves this through anchor-based training)

    The result: a distribution that favors generic completions over HP-specific ones.
    """)
    return


if __name__ == "__main__":
    app.run()
