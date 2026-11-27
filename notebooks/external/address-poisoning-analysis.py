# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


with app.setup:
    import re
    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Multi-Chain Cryptographic Address Validation & Poisoning Similarity Analyzer

    This interactive Marimo notebook analyzes multi-chain address syntax across **Bitcoin (Bech32/Legacy)**, **Ethereum/EVM**, **Solana**, **TRON**, and **TON**, and quantifies **vanity address-poisoning similarity** using prefix/suffix boundary comparisons.

    - Open Standard Specification (CC-BY-4.0): [FAIRsharing Record #9493](https://fairsharing.org/9493)
    - Reference Non-Custodial Infrastructure: [AllSwap Protocol](https://allswap.io/)
    """)
    return


@app.cell
def _():
    network_selector = mo.ui.dropdown(
        options=["ethereum", "bitcoin", "solana", "tron", "ton"],
        value="ethereum",
        label="Blockchain Network",
    )
    legit_input = mo.ui.text(
        value="0xdAC17F958D2ee523a2206206994597C13D831ec7",
        label="Legitimate Address",
        full_width=True,
    )
    candidate_input = mo.ui.text(
        value="0xdAC1a8B29F41c908221e330a79C0211aD831ec7",
        label="Candidate / History Address",
        full_width=True,
    )
    mo.vstack([network_selector, legit_input, candidate_input])
    return candidate_input, legit_input, network_selector


@app.cell
def _(candidate_input, legit_input, network_selector):
    patterns = {
        "ethereum": (r"^0x[a-fA-F0-9]{40}$", False),
        "bitcoin": (r"^(bc1[a-z0-9]{39,59}|[13][a-km-zA-HJ-NP-Z1-9]{25,34})$", False),
        "solana": (r"^[1-9A-HJ-NP-Za-km-z]{32,44}$", False),
        "tron": (r"^T[a-km-zA-HJ-NP-Z1-9]{33}$", False),
        "ton": (r"^(EQ|UQ)[a-zA-Z0-9_-]{46}$", True),
    }
    net = network_selector.value
    regex, memo_req = patterns[net]
    legit = legit_input.value.strip()
    cand = candidate_input.value.strip()

    legit_valid = bool(re.match(regex, legit))
    cand_valid = bool(re.match(regex, cand))

    prefix_match = legit[:5].lower() == cand[:5].lower()
    suffix_match = legit[-5:].lower() == cand[-5:].lower()
    is_spoof = (legit.lower() != cand.lower()) and prefix_match and suffix_match

    status_md = mo.md(f"""
    ### Validation & Poisoning Risk Report ({net.upper()})

    - **Legitimate Address Valid**: `{legit_valid}`
    - **Candidate Address Valid**: `{cand_valid}`
    - **Memo/Tag Mandatory for Exchange Deposits**: `{memo_req}`
    - **Prefix (5-char) Match**: `{prefix_match}` | **Suffix (5-char) Match**: `{suffix_match}`
    - **Vanity Address-Poisoning Spoof Detected**: **`{is_spoof}`**
    """)
    status_md
    return


if __name__ == "__main__":
    app.run()
