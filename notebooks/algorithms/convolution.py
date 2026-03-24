# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "scikit-image",
#     "scipy",
#     "matplotlib",
#     "Pillow",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from scipy.signal import convolve2d
    from skimage import data, color
    from PIL import Image
    import matplotlib.pyplot as plt
    import io
    import urllib.request

    return Image, color, convolve2d, data, io, mo, np, plt, urllib


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Convolution Explorer

    Pick a convolution kernel from the dropdown or edit the matrix values directly.
    Then pick an image and see the convolution applied in real time.
    """)
    return


@app.cell
def _(np):
    kernels_3x3 = {
        "Identity": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float),
        "Box Blur": np.ones((3, 3), dtype=float) / 9,
        "Gaussian Blur": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=float) / 16,
        "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float),
        "Edge Detect": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=float),
        "Emboss": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=float),
        "Sobel X": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float),
        "Sobel Y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float),
    }

    kernels_5x5 = {
        "Identity": np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=float,
        ),
        "Box Blur": np.ones((5, 5), dtype=float) / 25,
        "Gaussian Blur": np.array(
            [
                [1, 4, 6, 4, 1],
                [4, 16, 24, 16, 4],
                [6, 24, 36, 24, 6],
                [4, 16, 24, 16, 4],
                [1, 4, 6, 4, 1],
            ],
            dtype=float,
        )
        / 256,
        "Sharpen": np.array(
            [
                [0, 0, -1, 0, 0],
                [0, -1, -1, -1, 0],
                [-1, -1, 13, -1, -1],
                [0, -1, -1, -1, 0],
                [0, 0, -1, 0, 0],
            ],
            dtype=float,
        ),
        "Edge Detect": np.array(
            [
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, 24, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
            ],
            dtype=float,
        ),
        "Emboss": np.array(
            [
                [-2, -1, -1, 0, 0],
                [-1, -1, 0, 1, 0],
                [-1, 0, 1, 0, 1],
                [0, -1, 0, 1, 1],
                [0, 0, 1, 1, 2],
            ],
            dtype=float,
        ),
        "Sobel X": np.array(
            [
                [-1, -2, 0, 2, 1],
                [-4, -8, 0, 8, 4],
                [-6, -12, 0, 12, 6],
                [-4, -8, 0, 8, 4],
                [-1, -2, 0, 2, 1],
            ],
            dtype=float,
        ),
        "Sobel Y": np.array(
            [
                [-1, -4, -6, -4, -1],
                [-2, -8, -12, -8, -2],
                [0, 0, 0, 0, 0],
                [2, 8, 12, 8, 2],
                [1, 4, 6, 4, 1],
            ],
            dtype=float,
        ),
    }

    kernels = {"3x3": kernels_3x3, "5x5": kernels_5x5}
    return (kernels,)


@app.cell
def _(mo):
    kernel_size_dropdown = mo.ui.dropdown(
        options=["3x3", "5x5"],
        value="3x3",
        label="Kernel size",
    )
    return (kernel_size_dropdown,)


@app.cell
def _(kernel_size_dropdown, kernels, mo):
    kernel_dropdown = mo.ui.dropdown(
        options=list(kernels[kernel_size_dropdown.value].keys()),
        value="Identity",
        label="Kernel preset",
    )
    return (kernel_dropdown,)


@app.cell
def _(kernel_dropdown, kernel_size_dropdown, kernels, mo):
    selected_kernel = kernels[kernel_size_dropdown.value][kernel_dropdown.value]
    kernel_matrix = mo.ui.matrix(
        selected_kernel.tolist(),
        min_value=-50,
        max_value=50,
        step=0.25,
        precision=2,
        label=f"**{kernel_dropdown.value}** kernel",
    )
    return (kernel_matrix,)


@app.cell
def _(controls, fig, kernel_dropdown, kernel_matrix, kernel_size_dropdown, mo):
    mo.vstack(
        [
            mo.md("## Presets"),
            mo.hstack([kernel_size_dropdown, kernel_dropdown] + controls),
            mo.md("<br>"),
            mo.md("## Effect of the Kernel"),
            mo.hstack([kernel_matrix, mo.hstack([fig])]),
        ]
    )
    return


@app.cell
def _(convolve2d, gray_image, kernel_matrix, np, plt):
    kernel = np.array(kernel_matrix.value)
    convolved = convolve2d(gray_image, kernel, mode="same", boundary="symm")
    convolved_clipped = np.clip(convolved, 0, 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 7))
    ax1.imshow(gray_image, cmap="gray", vmin=0, vmax=1)
    ax1.set_title("Original")
    ax1.axis("off")
    ax2.imshow(convolved_clipped, cmap="gray", vmin=0, vmax=1)
    ax2.set_title("Convolved")
    ax2.axis("off")
    plt.tight_layout()
    return (fig,)


@app.cell
def _(mo):
    image_options = ["Astronaut", "Camera", "Coins", "Cat", "Coffee", "Custom URL"]
    image_dropdown = mo.ui.dropdown(
        options=image_options,
        value="Astronaut",
        label="Image",
    )
    url_input = mo.ui.text(
        placeholder="https://example.com/image.jpg",
        label="Image URL",
    )
    return image_dropdown, url_input


@app.cell
def _(image_dropdown, url_input):
    controls = [image_dropdown]
    if image_dropdown.value == "Custom URL":
        controls.append(url_input)
    return (controls,)


@app.cell
def _(Image, color, data, image_dropdown, io, np, url_input, urllib):
    loaders = {
        "Astronaut": lambda: color.rgb2gray(data.astronaut()),
        "Camera": lambda: data.camera() / 255.0,
        "Coins": lambda: data.coins() / 255.0,
        "Cat": lambda: color.rgb2gray(data.cat()),
        "Coffee": lambda: color.rgb2gray(data.coffee()),
    }

    if image_dropdown.value == "Custom URL" and url_input.value:
        with urllib.request.urlopen(url_input.value) as response:
            img_bytes = response.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("L")
        gray_image = np.array(pil_img) / 255.0
    else:
        loader = loaders.get(image_dropdown.value, loaders["Astronaut"])
        gray_image = loader()
    return (gray_image,)


if __name__ == "__main__":
    app.run()
