# /// script
# requires-python = "==3.12"
# dependencies = [
#     "marimo",
#     "numba==0.60.0",
#     "numpy==2.0.2",
#     "scikit-image==0.24.0",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import time
    import urllib.request
    import numpy as np
    from numba import jit
    from pathlib import Path
    from skimage import io, filters, transform


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Seam Carving

    _Example adapted from work by [Vincent Warmerdam](https://x.com/fishnets88)_.

    ## The seam carving algorithm
    This marimo demonstration is partially an homage to [a great video by Grant
    Sanderson](https://www.youtube.com/watch?v=rpB6zQNsbQU) of 3Blue1Brown, which demonstrates
    the seam carving algorithm in [Pluto.jl](https://plutojl.org/):

    <iframe width="560" height="315" src="https://www.youtube.com/embed/rpB6zQNsbQU?si=oiZclGIj2atJR47m" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

    As Grant explains, the seam carving algorithm preserves the shapes of the main content in the image, while killing the "dead space": the image is resized, but the clocks and other content are not resized or deformed.

    This notebook is a Python version of the seam carving algorithm, but it is also a
    demonstration of marimo's [caching
    feature](https://docs.marimo.io/guides/best_practices/performance.html#cache-computations-with-mo-cache),
    which is helpful because the algorithm is compute intensive even when you
    use [Numba](https://numba.pydata.org/).

    Try it out by playing with the slider!
    """)
    return


@app.cell(hide_code=True)
def _():
    input_image = "The_Persistence_of_Memory.jpg"
    if not Path(input_image).exists():
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/marimo-team/gallery-examples/main/notebooks/math/seam-carving/The_Persistence_of_Memory.jpg",
            input_image,
        )
    return (input_image,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Try it!
    """)
    return


@app.cell
def _():
    slider = mo.ui.slider(
        0.7,
        1.0,
        step=0.05,
        value=1.0,
        label="Amount of resizing to perform:",
        show_value=True,
    )
    slider
    return (slider,)


@app.cell
def _(input_image, slider):
    scale_factor = slider.value
    result = efficient_seam_carve(input_image, scale_factor)

    mo.hstack([mo.image(input_image), mo.image(result)], justify="start")
    return


@app.function
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


@app.function
def compute_energy_map(gray):
    return np.abs(filters.sobel_h(gray)) + np.abs(filters.sobel_v(gray))


@app.function
@jit(nopython=True)
def find_seam(energy_map):
    height, width = energy_map.shape
    dp = energy_map.copy()
    backtrack = np.zeros((height, width), dtype=np.int32)

    for i in range(1, height):
        for j in range(width):
            if j == 0:
                idx = np.argmin(dp[i - 1, j : j + 2])
                backtrack[i, j] = idx + j
                min_energy = dp[i - 1, idx + j]
            elif j == width - 1:
                idx = np.argmin(dp[i - 1, j - 1 : j + 1])
                backtrack[i, j] = idx + j - 1
                min_energy = dp[i - 1, idx + j - 1]
            else:
                idx = np.argmin(dp[i - 1, j - 1 : j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = dp[i - 1, idx + j - 1]

            dp[i, j] += min_energy

    return backtrack


@app.function
@jit(nopython=True)
def remove_seam(image, backtrack):
    height, width, _ = image.shape
    output = np.zeros((height, width - 1, 3), dtype=np.uint8)
    j = np.argmin(backtrack[-1])

    for i in range(height - 1, -1, -1):
        for k in range(3):
            output[i, :, k] = np.delete(image[i, :, k], j)
        j = backtrack[i, j]

    return output


@app.function
def seam_carving(image, new_width):
    height, width, _ = image.shape

    while width > new_width:
        gray = rgb2gray(image)
        energy_map = compute_energy_map(gray)
        backtrack = find_seam(energy_map)
        image = remove_seam(image, backtrack)
        width -= 1

    return image


@app.function
@mo.cache
def efficient_seam_carve(image_path, scale_factor):
    img = io.imread(image_path)
    new_width = int(img.shape[1] * scale_factor)

    start_time = time.time()
    carved_img = seam_carving(img, new_width)
    end_time = time.time()

    print(f"Seam carving completed in {end_time - start_time:.2f} seconds")

    return carved_img


if __name__ == "__main__":
    app.run()
