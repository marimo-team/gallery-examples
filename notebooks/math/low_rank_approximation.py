# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.8",
#     "numpy==2.4.3",
#     "scikit-image==0.26.0",
#     "wigglystuff==0.3.1",
# ]
# ///

import marimo

__generated_with = "0.22.0"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage import data
    from skimage.util import img_as_float
    from wigglystuff import WebcamCapture

    return WebcamCapture, mo, np, plt


@app.cell
def _(np):
    def low_rank_rgb(img, rank):
        height, width, channels = img.shape
        assert channels == 3, "Expected RGB image with shape (H, W, 3)"

        out = np.empty_like(img, dtype=np.float64)
        for c in range(3):
            channel = img[:, :, c].astype(np.float64)
            U, S, Vt = np.linalg.svd(channel, full_matrices=False)
            out[:, :, c] = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]

        return out

    return (low_rank_rgb,)


@app.cell(hide_code=True)
def _(WebcamCapture, mo):
    camera = mo.ui.anywidget(WebcamCapture())
    camera
    return (camera,)


@app.cell
def _(camera, mo, np):
    mo.stop(not camera.image_base64, mo.md("**Take a photo to get started.**"))

    pil_image = camera.get_pil()
    image = np.array(pil_image)[:, :, :3] / 255.0
    return (image,)


@app.cell
def _(image, low_rank_rgb, rank_slider):
    approx_image = low_rank_rgb(image, rank_slider.value)
    return (approx_image,)


@app.cell(hide_code=True)
def _(mo):
    rank_slider = mo.ui.slider(1, 128, value=20, label="Rank of approximation")
    rank_slider
    return (rank_slider,)


@app.cell(hide_code=True)
def _(approx_image, image, plt, rank_slider):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(approx_image)
    axes[1].set_title(f"Rank {rank_slider.value} Approximation")
    axes[1].axis("off")
    plt.tight_layout()
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
