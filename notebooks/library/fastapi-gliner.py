# /// script
# requires-python = "==3.12"
# dependencies = [
#     "marimo",
#     "fastapi==0.122.0",
#     "gliner2==1.0.2",
#     "ipython==9.7.0",
#     "pydantic==2.12.5",
#     "wigglystuff==0.2.0",
#     "anthropic==0.75.0",
#     "pytest==9.0.1",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium", sql_output="polars")

with app.setup:
    import marimo as mo
    import json
    import os
    import warnings
    import pytest
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from gliner2 import GLiNER2
    from pydantic import BaseModel
    from wigglystuff import SortableList

    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
    with mo.capture_stdout() as _buffer1:
        extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    class ModelInput(BaseModel):
        text: str
        entities: list[str]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # GliNER v2

    This notebook contains a special demo. It highlights a new version of GliNER, which historically mainly tried to focus on the named entitiy recognition usecase, but now also supports structured information parsing as well as classification. Think of it as a very lightweight LLM that can still do zero-shot tasks for NLP, but has the benefit of running locally and on a CPU.

    This notebook also highlights the flexibility of marimo. This notebook uses pydantic in a clever way so that we can run this one notebook as an interactive webapp, an API or a command-line utility. Neat!
    """)
    return


@app.function
def extract(model_input: ModelInput):
    return {
        "text": model_input.text,
        **extractor.extract_entities(model_input.text, model_input.entities),
    }


@app.cell
def _(entities, text_input):
    cli_args = mo.cli_args()

    if mo.app_meta().mode == "script":
        if "help" in cli_args or len(cli_args) == 0:
            print(
                "You can pass --text and --entities to this cli to extract entities. Note that entities needs to be a comma delimited sequence of topics."
            )
            exit()

    ents = mo.cli_args().get("entities", None)

    model_input = ModelInput(
        text=mo.cli_args().get("text", text_input.value),
        entities=ents.split(",") if ents else entities.value.get("value"),
    )
    return (model_input,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## UI Elements
    """)
    return


@app.cell
def _():
    text_input = mo.ui.text_area(rows=2)
    entities = mo.ui.anywidget(
        SortableList(["location", "person"], addable=True, removable=True, editable=True)
    )
    return entities, text_input


@app.cell
def _(entities, out, text_input):
    mo.vstack(
        [
            mo.md("### text input"),
            text_input,
            mo.md("### entities"),
            entities,
            mo.md(f"""
    ```
    {json.dumps(out, indent=2)}
    ```
    """)
        ]
    )
    return


@app.cell(hide_code=True)
def _(model_input):
    out = extract(model_input)

    if mo.app_meta().mode == "script":
        print(json.dumps(out))
    return (out,)


@app.function
def create_app():
    fastapi_app = FastAPI()

    @fastapi_app.post("/extract/")
    async def extract_web(model_input: ModelInput):
        return extract(model_input=model_input)

    @fastapi_app.get("/")
    @fastapi_app.get("/health/")
    @fastapi_app.get("/healthz/")
    async def health():
        return {"status": "alive"}

    return fastapi_app


@app.cell
def _():
    client = TestClient(create_app())
    return (client,)


@app.cell
def _(client):
    @pytest.mark.parametrize("route", ["/", "health", "healthz"])
    def test_read_main(route):
        response = client.get(route)
        assert response.status_code == 200


    def test_smoke():
        response = client.post(
            "/extract/", json={"text": "Vincent has a happy cat", "entities": ["animal", "person"]}
        )

        model_out = response.json()["entities"]
        assert model_out["animal"] == ["cat"]
        assert model_out["person"] == ["Vincent"]

    return


if __name__ == "__main__":
    app.run()
