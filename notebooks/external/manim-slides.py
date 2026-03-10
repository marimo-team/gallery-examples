# /// script
# requires-python = "==3.12"
# dependencies = [
#     "manim==0.19.1",
#     "manim-slides==5.5.2",
#     "marimo>=0.20.4",
#     "mohtml==0.1.11",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium", sql_output="polars")

with app.setup:
    from manim import (
        Dot,
        Circle,
        Square,
        VGroup,
        Text,
        MathTex,
        Axes,
        FadeIn,
        FadeOut,
        Create,
        Write,
        Transform,
        ReplacementTransform,
        MoveToTarget,
        AnimationGroup,
        Succession,
        BLUE,
        RED,
        GREEN,
        WHITE,
        YELLOW,
        BLACK,
        ORIGIN,
        UP,
        DOWN,
        LEFT,
        RIGHT,
        linear,
        smooth,
        config,
        TexTemplate
    )
    from manim_slides import Slide
    import numpy as np


@app.cell
def _():
    import marimo as mo
    from pathlib import Path

    return Path, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is a demonstration of [manim-slides](https://manim-slides.eertmans.be/latest/index.html) via a marimo notebook.

    ## The math

    > This is the math that we will visualise with manim slides in a moment.

    We want to simplify this:

    $$
    \frac{\sin^2 x}{1 - \cos x} - 1
    $$

    This becomes.

    $$
    = \frac{1 - \cos^2 x}{1 - \cos x} - 1
    $$

    You can factor this.

    $$
    = \frac{(1 - \cos x)(1 + \cos x)}{1 - \cos x} - 1
    $$

    And hey! Notice how things cancel out pretty early here!

    $$
    = \frac{(\cancel{1 - \cos x})(1 + \cos x)}{\cancel{1 - \cos x}} - 1
    $$

    That makes things a lot simpler.

    $$
    = 1 + \cos x - 1
    $$

    $$
    = \cos x
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The code

    What follows next is some code that was generated with Claude. The cool thing here is that you can use marimo to host the markdown, then point Claude at the notebook to read the math, and then get Claude to add the relevant slides to the same notebook.
    """)
    return


@app.class_definition
class SimpleSlides(Slide):
    def construct(self):
        # Add cancel package to LaTeX preamble
        tex_template = TexTemplate()
        tex_template.add_to_preamble(r"\usepackage{cancel}")
        MathTex.set_default(tex_template=tex_template)

        # Set black background with white text
        self.camera.background_color = BLACK

        # Title slide
        title = Text("Simplifying a Trigonometric Expression", font_size=40, color=WHITE)
        self.play(Write(title))
        self.next_slide()
        self.play(FadeOut(title))

        # Step 1: Original formula (centered)
        comment1 = Text("We want to simplify this:", font_size=24, color=WHITE).to_edge(UP, buff=0.5)
        formula1 = MathTex("\\frac{\\sin^2 x}{1 - \\cos x} - 1", font_size=44, color=WHITE)
        self.play(Write(comment1))
        self.play(Write(formula1))
        self.next_slide()

        # Step 2: formula1 = formula2 (keep equals centered)
        comment2 = Text("Use the identity: sin²x = 1 - cos²x", font_size=24, color=WHITE).to_edge(UP, buff=0.5)

        equals1 = MathTex("=", font_size=44, color=WHITE)
        formula2 = MathTex("\\frac{1 - \\cos^2 x}{1 - \\cos x} - 1", font_size=44, color=WHITE)

        # Position formula2 to the right of equals, then move equals to center
        formula2.next_to(equals1, RIGHT, buff=0.3)
        equals1.move_to(ORIGIN)
        formula2.next_to(equals1, RIGHT, buff=0.3)

        self.play(Transform(comment1, comment2))
        self.play(formula1.animate.next_to(equals1, LEFT, buff=0.3))
        self.play(Write(equals1))
        self.play(Write(formula2))
        self.next_slide()

        # Step 3: formula2 = formula3 (keep equals centered, fade out formula1)
        comment3 = Text("Factor the numerator:", font_size=24, color=WHITE).to_edge(UP, buff=0.5)

        equals2 = MathTex("=", font_size=44, color=WHITE).move_to(ORIGIN)
        formula3 = MathTex("\\frac{(1 - \\cos x)(1 + \\cos x)}{1 - \\cos x} - 1", font_size=38, color=WHITE).next_to(equals2, RIGHT, buff=0.3)

        self.play(
            FadeOut(formula1),
            FadeOut(equals1),
            formula2.animate.next_to(equals2, LEFT, buff=0.3),
            Transform(comment1, comment3)
        )
        self.play(Write(equals2))
        self.play(Write(formula3))
        self.next_slide()

        # Step 4: formula3 = formula3_cancelled (show cancellation)
        comment4 = Text("Cancel the common terms!", font_size=24, color=WHITE).to_edge(UP, buff=0.5)

        equals3 = MathTex("=", font_size=44, color=WHITE).move_to(ORIGIN)
        formula3_cancelled = MathTex(
            "\\frac{(\\cancel{1 - \\cos x})(1 + \\cos x)}{\\cancel{1 - \\cos x}} - 1", 
            font_size=38, 
            color=WHITE
        ).next_to(equals3, RIGHT, buff=0.3)

        self.play(
            FadeOut(formula2),
            FadeOut(equals2),
            formula3.animate.next_to(equals3, LEFT, buff=0.3),
            Transform(comment1, comment4)
        )
        self.play(Write(equals3))
        self.play(Write(formula3_cancelled))
        self.next_slide()

        # Step 5: formula3_cancelled = formula4
        comment5 = Text("Simplify:", font_size=24, color=WHITE).to_edge(UP, buff=0.5)

        equals4 = MathTex("=", font_size=44, color=WHITE).move_to(ORIGIN)
        formula4 = MathTex("1 + \\cos x - 1", font_size=44, color=WHITE).next_to(equals4, RIGHT, buff=0.3)

        self.play(
            FadeOut(formula3),
            FadeOut(equals3),
            formula3_cancelled.animate.next_to(equals4, LEFT, buff=0.3),
            Transform(comment1, comment5)
        )
        self.play(Write(equals4))
        self.play(Write(formula4))
        self.next_slide()

        # Step 6: formula4 = formula5 (final answer)
        comment6 = Text("Final result:", font_size=24, color=WHITE).to_edge(UP, buff=0.5)

        equals5 = MathTex("=", font_size=44, color=WHITE).move_to(ORIGIN)
        formula5 = MathTex("\\cos x", font_size=50, color=WHITE).next_to(equals5, RIGHT, buff=0.3)

        self.play(
            FadeOut(formula3_cancelled),
            FadeOut(equals4),
            formula4.animate.next_to(equals5, LEFT, buff=0.3),
            Transform(comment1, comment6)
        )
        self.play(Write(equals5))
        self.play(Write(formula5))
        self.next_slide()

        # Final slide: Show complete result
        final_comment = Text("Complete proof:", font_size=24, color=WHITE).to_edge(UP, buff=0.5)
        final_formula = MathTex("\\frac{\\sin^2 x}{1 - \\cos x} - 1 = \\cos x", font_size=48, color=WHITE)
        self.play(
            FadeOut(formula4),
            FadeOut(equals5),
            FadeOut(formula5),
            Transform(comment1, final_comment)
        )
        self.play(Write(final_formula))
        self.next_slide()


@app.cell(hide_code=True)
def _():
    import subprocess

    SimpleSlides

    subprocess.run(f"manim-slides render {__file__} SimpleSlides", shell=True, check=True)
    return (subprocess,)


@app.cell
def _(subprocess):
    subprocess.run("manim-slides convert SimpleSlides -c controls=true simple.html --one-file", shell=True, check=True)
    return


@app.cell
def _(Path, mo):
    mo.iframe(Path("simple.html").read_text())
    return


if __name__ == "__main__":
    app.run()
