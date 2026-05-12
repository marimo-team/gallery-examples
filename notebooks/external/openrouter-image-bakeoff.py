# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.23.2,<0.23.6",
#     "openai>=1.50.0",
#     "wigglystuff>=0.2.39",
#     "pillow>=10.0.0",
# ]
# ///

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import asyncio
    import base64
    import io
    import time

    import marimo as mo
    from openai import AsyncOpenAI, OpenAI
    from PIL import Image, ImageOps
    from wigglystuff import EnvConfig, Paint

    return (
        AsyncOpenAI,
        EnvConfig,
        Image,
        ImageOps,
        OpenAI,
        Paint,
        asyncio,
        base64,
        io,
        mo,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # OpenRouter: image-model bake-off

    [OpenRouter](https://openrouter.ai/) now exposes [image generation](https://openrouter.ai/collections/image-models) through its chat-completions API by
    passing `extra_body={"modalities": ["image"]}`. That means we can compare
    image output across very different providers using one API surface and one
    key.

    This notebook lets you pick **three** image models, type a prompt, and
    optionally sketch a reference, then see the results side-by-side. When you
    like one of the results, promote it to **stage 2** to draw annotations on
    top and iterate. The three model calls run concurrently via
    `asyncio.gather`, so wall time is roughly the slowest model, not the sum.
    """)
    return


@app.cell(hide_code=True)
def _(EnvConfig, OpenAI, mo):
    def check_openrouter_key(k):
        OpenAI(base_url="https://openrouter.ai/api/v1", api_key=k).models.list()

    env_config = mo.ui.anywidget(
        EnvConfig({"OPENROUTER_API_KEY": check_openrouter_key})
    )
    env_config
    return (env_config,)


@app.cell(hide_code=True)
def _(env_config):
    env_config.require_valid()
    return


@app.cell(hide_code=True)
def _(mo):
    MODELS = [
        "bytedance-seed/seedream-4.5",
        "google/gemini-2.5-flash-image",
        "black-forest-labs/flux.2-pro",
        "black-forest-labs/flux.2-klein-4b",
        "openai/gpt-5-image-mini",
        "openai/gpt-5-image",
        "stability-ai/stable-diffusion-3.5-large",
    ]

    model_a = mo.ui.dropdown(MODELS, value=MODELS[0], label="Model A")
    model_b = mo.ui.dropdown(MODELS, value=MODELS[1], label="Model B")
    model_c = mo.ui.dropdown(MODELS, value=MODELS[2], label="Model C")
    return model_a, model_b, model_c


@app.cell(hide_code=True)
def _(Paint, mo):
    ASPECTS = {
        "1:1": {"api": "1024x1024", "display": (400, 400)},
        "16:9": {"api": "1280x720", "display": (480, 270)},
        "9:16": {"api": "720x1280", "display": (270, 480)},
        "4:3": {"api": "1152x864", "display": (400, 300)},
        "3:4": {"api": "864x1152", "display": (300, 400)},
    }

    prompt = mo.ui.text_area(
        value="A beautiful sunset over mountains",
        label="Prompt",
        rows=3,
        full_width=True,
    )
    paint = mo.ui.anywidget(Paint(width=320, height=200))
    aspect = mo.ui.dropdown(list(ASPECTS.keys()), value="1:1", label="Aspect")
    button = mo.ui.run_button(label="Generate")
    return ASPECTS, aspect, button, paint, prompt


@app.cell(hide_code=True)
def _(aspect, button, mo, model_a, model_b, model_c, paint, prompt):
    mo.vstack(
        [
            mo.md("## Stage 1: Generate"),
            paint,
            mo.hstack([model_a, model_b, model_c], justify="start"),
            prompt,
            mo.hstack([aspect, button], justify="start"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(Image, header, mo, results, use_buttons):
    _columns = [
        mo.vstack(
            [
                header(name, elapsed, cost),
                content if isinstance(content, Image.Image) else mo.md(f"_{content}_"),
                use_buttons[idx],
            ],
            gap=0.25,
        )
        for idx, (name, content, cost, elapsed) in enumerate(results)
    ]
    mo.hstack(_columns, widths="equal")
    return


@app.cell(hide_code=True)
def _(ASPECTS, AsyncOpenAI, Image, ImageOps, base64, env_config, io, mo, time):
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=env_config.widget["OPENROUTER_API_KEY"],
    )

    def image_to_data_url(pil_img):
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{b64}"

    def paint_to_data_url(pil_img):
        if pil_img is None:
            return None
        rgb = pil_img.convert("RGB")
        extrema = rgb.getextrema()
        if all(lo == 255 and hi == 255 for lo, hi in extrema):
            return None
        return image_to_data_url(pil_img)

    async def generate_image(model, prompt_text, aspect_key, image_data_url=None):
        spec = ASPECTS[aspect_key]
        if image_data_url:
            content = [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]
        else:
            content = prompt_text
        start = time.perf_counter()
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                extra_body={
                    "modalities": ["image"],
                    "usage": {"include": True},
                    "size": spec["api"],
                },
            )
            elapsed = time.perf_counter() - start
            usage = response.usage
            cost = getattr(usage, "cost", None)
            if cost is None and usage is not None:
                cost = (getattr(usage, "model_extra", None) or {}).get("cost")
            message = response.choices[0].message
            images = getattr(message, "images", None) or []
            if not images:
                return f"No image returned. Text reply: {message.content!r}", cost, elapsed
            url = images[0]["image_url"]["url"]
            if url.startswith("data:"):
                _, b64 = url.split(",", 1)
                img = Image.open(io.BytesIO(base64.b64decode(b64)))
                img = ImageOps.fit(img, spec["display"], Image.LANCZOS)
                return img, cost, elapsed
            return url, cost, elapsed
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}", None, time.perf_counter() - start

    def header(name, elapsed, cost):
        cost_str = f"${cost:.4f}" if cost is not None else "cost n/a"
        return mo.Html(
            f'<div style="line-height:1.15;margin:0;padding:0;">'
            f'<div style="font-weight:600;">{name}</div>'
            f'<div style="font-size:0.8em;color:#888;">'
            f'{elapsed:.1f}s &middot; {cost_str}'
            f'</div></div>'
        )

    return generate_image, header, image_to_data_url, paint_to_data_url


@app.cell
def _(mo):
    selected_image, set_selected_image = mo.state(None)
    return selected_image, set_selected_image


@app.cell(hide_code=True)
async def _(
    Image,
    aspect,
    asyncio,
    button,
    generate_image,
    mo,
    model_a,
    model_b,
    model_c,
    paint,
    paint_to_data_url,
    prompt,
):
    mo.stop(
        not button.value,
        mo.md("*Pick three models, write a prompt (optionally sketch a reference), then click **Generate**.*"),
    )

    image_data_url = paint_to_data_url(paint.get_pil())
    selections = [model_a.value, model_b.value, model_c.value]
    raw = await asyncio.gather(
        *[generate_image(name, prompt.value, aspect.value, image_data_url) for name in selections]
    )
    results = [(name, *r) for name, r in zip(selections, raw)]
    images = [r[1] if isinstance(r[1], Image.Image) else None for r in results]
    return images, results


@app.cell(hide_code=True)
def _(images, mo, set_selected_image):
    def make_cb(img):
        def cb(v):
            if img is not None:
                set_selected_image(img)
            return v + 1
        return cb

    use_buttons = mo.ui.array(
        [
            mo.ui.button(
                value=0,
                on_click=make_cb(img),
                label="Use this →",
                disabled=img is None,
            )
            for img in images
        ]
    )
    return (use_buttons,)


@app.cell(hide_code=True)
def _(mo):
    prompt2 = mo.ui.text_area(
        value="Improve this image, following the annotations drawn on top.",
        label="Refinement prompt",
        rows=2,
        full_width=True,
    )
    button2 = mo.ui.run_button(label="Generate refinements")
    return button2, prompt2


@app.cell(hide_code=True)
def _(mo, selected_image):
    if selected_image() is None:
        header2 = mo.md(
            "---\n\n"
            "## Stage 2: Refine\n\n"
            "*Click **Use this →** under any image above to bring it here.*"
        )
    else:
        header2 = mo.md(
            "---\n\n"
            "## Stage 2: Refine\n\n"
            "*Sketch annotations on top of the image below, then generate.*"
        )
    header2
    return


@app.cell(hide_code=True)
def _(Paint, mo, selected_image):
    sel = selected_image()
    if sel is None:
        paint2 = None
        out = None
    else:
        paint2 = mo.ui.anywidget(Paint(init_image=sel, store_background=True))
        out = paint2
    out
    return (paint2,)


@app.cell(hide_code=True)
def _(aspect, button2, mo, model_a, model_b, model_c, paint2, prompt2):
    if paint2 is None:
        controls2 = None
    else:
        controls2 = mo.vstack(
            [
                mo.hstack([model_a, model_b, model_c], justify="start"),
                prompt2,
                mo.hstack([aspect, button2], justify="start"),
            ]
        )
    controls2
    return


@app.cell(hide_code=True)
async def _(
    Image,
    aspect,
    asyncio,
    button2,
    generate_image,
    image_to_data_url,
    mo,
    model_a,
    model_b,
    model_c,
    paint2,
    prompt2,
):
    mo.stop(paint2 is None, None)
    mo.stop(
        not button2.value,
        mo.md("*Draw on the image and click **Generate refinements**.*"),
    )

    refined_url = image_to_data_url(paint2.get_pil())
    selections2 = [model_a.value, model_b.value, model_c.value]
    raw2 = await asyncio.gather(
        *[generate_image(name, prompt2.value, aspect.value, refined_url) for name in selections2]
    )
    results2 = [(name, *r) for name, r in zip(selections2, raw2)]
    images2 = [r[1] if isinstance(r[1], Image.Image) else None for r in results2]
    return images2, results2


@app.cell(hide_code=True)
def _(images2, mo, set_selected_image):
    def make_cb2(img):
        def cb(v):
            if img is not None:
                set_selected_image(img)
            return v + 1
        return cb

    use_buttons2 = mo.ui.array(
        [
            mo.ui.button(
                value=0,
                on_click=make_cb2(img),
                label="Use this →",
                disabled=img is None,
            )
            for img in images2
        ]
    )
    return (use_buttons2,)


@app.cell(hide_code=True)
def _(Image, header, mo, results2, use_buttons2):
    _columns2 = [
        mo.vstack(
            [
                header(name, elapsed, cost),
                content if isinstance(content, Image.Image) else mo.md(f"_{content}_"),
                use_buttons2[idx],
            ],
            gap=0.25,
        )
        for idx, (name, content, cost, elapsed) in enumerate(results2)
    ]
    mo.hstack(_columns2, widths="equal")
    return


if __name__ == "__main__":
    app.run()
