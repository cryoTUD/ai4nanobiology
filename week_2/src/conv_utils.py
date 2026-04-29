# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 16:23:05 2020

@author: abharadwaj1
"""

# imports
from time import time
import numpy as np
import mrcfile

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.signal.windows import gaussian
from scipy.signal import convolve2d
from skimage.color import rgb2gray
from skimage.transform import resize
from src.filters import (
    sobel,
    sharpen,
    box_blur,
    gaussian_blur,
)
from IPython.display import display
from ipysheet import to_array
from ipywidgets import (
    interact,
    IntSlider,
    fixed,
    interact_manual,
    HBox, VBox, Layout, Output, Label
)


# functions
def getbox(width, N=1000, index=500, high=1, low=0):
    """returns a box function of lenght N and box size a
    Note: signal[mid-halfbox] = high
          signal[mid+halfbox] = low
    """
    signal = np.zeros(N) + low
    halfbox = int(width / 2)
    signal[index - halfbox : index + halfbox + 1] = high
    return signal


def get_image(name, size):
    if name == "cameraman":
        im_in = rgb2gray(plt.imread("data/cameraman.bmp"))
        im_out = resize(im_in, (size, size))

    if name == "thankyou":
        im_in = rgb2gray(plt.imread("data/thankyou.jpg"))
        im_out = resize(im_in, (size, size))

    if name == "cryo_em_image":
        mrc_in = np.flip(
            mrcfile.open("data/6nbcmicrograph_0_c0_-2502.mrc").data, axis=0
        )[0]
        # crop the central 1024x1024 pixels from original image
        mrc_in = mrc_in[512:1536, 512:1536]
        # im_in = rgb2gray(mrc_in)
        im_out = resize(mrc_in, (size, size))
        im_out = np.clip(
            im_out,
            np.mean(im_out) - 2 * np.std(im_out),
            np.mean(im_out) + 2 * np.std(im_out),
        )

    if name == "spike_protein":
        mrc_in = np.flip(
            mrcfile.open(
                "data/pdb_6crv_removed_glycan_"
                + "atoms_partialmicrograph_0_c0_-50000.mrc"
            ).data,
            axis=0,
        )[0]
        # crop the central 1024x1024 pixels from original image
        mrc_in = mrc_in[512:1536, 512:1536]
        # im_in = rgb2gray(mrc_in)
        im_out = resize(mrc_in, (size, size))

    if name == "tudelft":
        im_out = np.flip(mrcfile.open("data/tud_flame.mrc").data, axis=0)
        im_out = resize(im_out, (size, size), anti_aliasing=True)

    return im_out


# functions
def convolution_illustrate(index, type):
    index = 500 + index
    if type == "box":
        box1 = getbox(width=300, N=1000, index=500, high=1, low=0)
        box2 = getbox(width=150, N=1000, index=index, high=0.6, low=0)
        box2_static = getbox(width=150, N=1000, index=500, high=0.6, low=0)
        convolution = np.convolve(box1, box2_static, mode="same") / 1000
    elif type == "gaussian":
        box1 = gaussian(1000, std=150)
        box2 = gaussian(10000, std=75)
        cropped_length = 1000
        peak_position_after_cropping = index
        crop_index_start = 5000 - peak_position_after_cropping
        crop_index_end = crop_index_start + cropped_length
        box2 = box2[crop_index_start:crop_index_end]
        box2_static = gaussian(1000, std=75)
        convolution = np.convolve(box1, box2_static, mode="same") / 1000

    product = box1 * box2
    product_sum = np.sum(product) / 1000
    xaxis = np.linspace(-500, 500, 1000)
    plt.plot(xaxis, box1, label="$f(x)$")
    plt.plot(xaxis, box2, label="$g(x)$")
    plt.plot(xaxis, convolution, label="$f(x) * g(x)$")
    plt.plot(index - 500, product_sum, "ro")
    # plot the legend next to the plot (outside the plot)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.xlabel("x")
    plt.show()


def play_with_convolution_illustration():
    return interact(
        convolution_illustrate,
        index=IntSlider(
            description="x_i", min=-400, max=400, step=1, value=150
        ),
        type=["box", "gaussian"],
    )

def play_with_2d_convolution():
    return interact(
        convolution_2d_illustrate,
        x_index=IntSlider(
            description="x_i", min=0, max=400, step=1, value=150
        ),
        y_index=IntSlider(
            description="y_i", min=0, max=400, step=1, value=150
        ),
        type=["blur", "horizontal_edge", "sharpen", "vertical_edge", "waldo"],
    )

def convolution_2d_illustrate(x_index, y_index, type):
    if type == "blur":
        kernel = box_blur(25)
    elif type == "horizontal_edge":
        kernel = sobel(25, "x")
    elif type == "vertical_edge":
        kernel = sobel(25, "y")
    elif type == "sharpen":
        kernel = sharpen(25)
    elif type == "waldo":
        # Alternating horizontal stripes — Waldo's shirt in grayscale is
        # bright / dark / bright / dark rows. Five stripes, each `sh` rows
        # tall and `sw` columns wide. Normalised so the response is
        # comparable in magnitude to the other kernels.
        sh, sw, n_stripes = 5, 25, 5
        kernel = np.vstack([
            ((-1) ** i) * np.ones((sh, sw), dtype=np.float32)
            for i in range(n_stripes)
        ])
        kernel /= np.abs(kernel).sum()

    # --- visualisation: show kernel sitting on the image at (x_index, y_index)
    Image = get_image("waldo" if type == "waldo" else "cameraman", size=400)

    full_response = convolve2d(Image, kernel, mode="same")

    kh, kw = kernel.shape
    y0, x0 = y_index - kh // 2, x_index - kw // 2
    y1, x1 = y0 + kh, x0 + kw

    # Local patch under the kernel and its pointwise product
    patch = np.zeros_like(kernel)
    ys, xs = max(y0, 0), max(x0, 0)
    ye, xe = min(y1, Image.shape[0]), min(x1, Image.shape[1])
    patch[ys - y0:ye - y0, xs - x0:xe - x0] = Image[ys:ye, xs:xe]
    product = patch * kernel
    response_here = product.sum()

    # Image with kernel footprint highlighted, plus kernel, product, output
    overlay = np.stack([Image] * 3, axis=-1)
    overlay = overlay / overlay.max()
    overlay[ys:ye, xs:xe, 0] = 1.0  # red box where the kernel sits

    f, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(overlay);            axes[0].set_title("image + kernel position")
    axes[1].imshow(kernel, cmap="gray"); axes[1].set_title("kernel")
    axes[2].imshow(product, cmap="gray"); axes[2].set_title(f"f·g, sum = {response_here:.3f}")
    axes[3].imshow(full_response, cmap="gray"); axes[3].set_title("full convolution")
    for ax in axes: ax.axis("off")
    plt.show()
    
def play_with_image_kernels(Image):
    dict_ktype = [
        ("Sobel", "sobel"),
        ("Sharpen", "sharpen"),
        ("Box blur", "box"),
        ("Gaussian blur", "gaussian"),
    ]
    interact_manual(
        plot_image_kernel_conv,
        Image=fixed(Image),
        kerneltype=dict_ktype,
        size=IntSlider(value=3, min=3, max=51, step=2),
        inputkernel=fixed(None),
    )


def plot_image_kernel_conv(Image, kerneltype, size, inputkernel=None):
    if kerneltype == "sobel":
        sobelx, sobely = sobel(size, "xy")
        output_dx = convolve2d(Image, sobelx, mode="same")
        output_dy = convolve2d(Image, sobely, mode="same")
        output = np.sqrt(output_dx**2 + output_dy**2)
        df_dx = pd.DataFrame(sobelx)
        df_dy = pd.DataFrame(sobely)
        print("Sobel derivative in X: \n")
        display(df_dx)
        print("Sobel derivative in Y: \n")
        display(df_dy)
        print(
            "Results cw from top-left: input image, sobel x, output, sobel y"
        )
        show_these((Image, output_dx, output_dy, output))

    elif kerneltype == "custom":
        kernel = inputkernel
    else:
        kernel = get_kernel_from_type(kerneltype, size)

    if kerneltype != "sobel":
        output = convolve2d(Image, kernel, mode="same")
        kernel_df = pd.DataFrame(kernel)
        print("The kernel used is: \n")
        display(kernel_df)
        print("Input image and after convolution")
        show_these((Image, output))


def show_these(allplots, fsize=12):
    f = plt.figure(figsize=(fsize, fsize))
    n = len(allplots)
    k = 1
    num_cols = 2
    num_rows = n // 2 + 1
    plt.subplots_adjust(wspace=1 / n, hspace=1 / n)
    for im in allplots:
        subplot_index = num_rows * 100 + num_cols * 10 + k
        ax = f.add_subplot(subplot_index)
        ax.imshow(im, cmap="gray")
        k += 1


def get_kernel_from_type(kerneltype, size):
    if kerneltype == "sharpen":
        kernel = sharpen(size)
    elif kerneltype == "box":
        kernel = box_blur(size)
    elif kerneltype == "gaussian":
        kernel = gaussian_blur(size)

    return kernel


def extract_kernel_and_convolve(Image, isheet):
    kernel = to_array(isheet)
    plot_image_kernel_conv(Image, "custom", isheet.rows, kernel)


def convolution_theorem(Image):
    dict_ktype = [
        ("Sharpen", "sharpen"),
        ("Box blur", "box"),
        ("Gaussian blur", "gaussian"),
    ]
    interact_manual(
        demonstrate_conv_theorem,
        Image=fixed(Image),
        kerneltype=dict_ktype,
        size=IntSlider(value=3, min=3, max=51, step=2),
        justshow=fixed(True),
    )


def demonstrate_conv_theorem(Image, kerneltype, size, justshow=True):
    kernel = get_kernel_from_type(kerneltype, size)
    kernel = get_padded_kernel(kernel, Image.shape)
    convoluted = conv_with_mask_using_fft(Image, kernel)
    if justshow:
        show_these(
            (
                Image,
                kernel,
                np.log(abs(np.fft.fftshift(np.fft.fft2(Image))) / Image.size),
                np.log(abs(np.fft.fftshift(np.fft.fft2(kernel))) / Image.size),
                convoluted,
            )
        )
    else:
        return convoluted


def get_padded_kernel(kernel, shape):
    bigkernel = np.zeros(shape)
    kwidth, kheight = kernel.shape
    istart, jstart = shape[0] // 2 - kheight // 2, shape[1] // 2 - kwidth // 2
    bigkernel[istart : istart + kheight, jstart : jstart + kwidth] = kernel
    return bigkernel


def conv_with_mask_using_fft(i, mask):
    ft_i = np.fft.fftshift(np.fft.fft2(i))
    ft_mask = np.fft.fftshift(np.fft.fft2(mask))
    ft_conv = ft_i * ft_mask
    conv = np.fft.ifftshift(np.fft.ifft2(ft_conv))
    conv = abs(conv) / conv.size
    return conv


def conv_with_mask_using_conv2d(i, mask):
    conv = convolve2d(i, mask, mode="same")
    return conv


def compare_fft_convolve(Image):
    dict_ktype = [
        ("Sharpen", "sharpen"),
        ("Box blur", "box"),
        ("Gaussian blur", "gaussian"),
    ]
    interact_manual(
        get_timings_convolution_theorem,
        Image=fixed(Image),
        kerneltype=dict_ktype,
        size=IntSlider(value=3, min=3, max=51, step=2),
    )


def get_timings_convolution_theorem(Image, kerneltype, size):
    tic = time()
    fft_convoluted = demonstrate_conv_theorem(
        Image, kerneltype, size, justshow=False
    )
    time_fft_convoluted = time() - tic

    toc = time()
    kernel = get_kernel_from_type(kerneltype, size)
    conv2d_convoluted = conv_with_mask_using_conv2d(Image, kernel)
    time_conv2d_convoluted = time() - toc

    print(
        "Time take for FFT algorithm: "
        + str(time_fft_convoluted)
        + " s \n"
        + "Time take for Convolution algorithm: "
        + str(time_conv2d_convoluted)
        + " s \n"
    )
    show_these((Image, fft_convoluted, Image, conv2d_convoluted))



def slide_kernel_over_image(image, user_kernel=None, size=15):
    """
    Interactive demo: pick a kernel, slide it over `image`, and watch the
    dot product (= one pixel of the convolution output) update live.

    Left   : image with the kernel's footprint outlined at (x, y).
    Middle : the chosen kernel as a heatmap.
    Right  : the pointwise product patch · kernel, with sum in the title.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from ipywidgets import interact, IntSlider, Dropdown
    from src.filters import sobel, sharpen, gaussian_blur
    from scipy.signal import convolve2d
    # Pre-build all kernels at the chosen size
    sx, sy = sobel(size, "xy")
    kernels = {
        "Vertical edge (Sobel x)":   sx,
        "Horizontal edge (Sobel y)": sy,
        "Gaussian blur":             gaussian_blur(size),
        "Sharpen":                   sharpen(size),
        "User kernel":                user_kernel if user_kernel is not None else np.zeros((size, size)),
    }

    H, W = image.shape[:2]
    half = size // 2
    normalize = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
    image = normalize(image)
    
    default_kernel_name = "Gaussian blur" if user_kernel is None else "User kernel"
    def update(kernel_name, x, y):
        # get full convolution by convolving the whole image with the kernel without scipy 
        full_convolution = np.zeros_like(image)
        kernel = kernels[kernel_name]
        if kernel_name not in ["Gaussian blur", "Sharpen"]:
            kernel = kernel / np.max(np.abs(kernel))
        kh, kw = kernel.shape
        for i in range(half, H - half):
            for j in range(half, W - half):
                y0, x0 = i - half, j - half
                patch = image[y0:y0 + kh, x0:x0 + kw]
                full_convolution[i, j] = (patch * kernel).sum()
        
        #full_convolution = normalize(full_convolution)
        pixel_val = full_convolution[y, x]
        # kernel = kernels[kernel_name]
        # normalise kernel 
        # kernel = normalize(kernel) * 2 - 1
        y0, x0 = y - half, x - half
        patch = image[y0:y0 + size, x0:x0 + size].astype(np.float32)
        product = patch * kernel
        result = product.sum()

        kmax = kernel.max() + 1e-9
        pmax = product.max() + 1e-9
        kmin = kernel.min() - 1e-9
        pmin = product.min() - 1e-9

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(image, cmap="gray")
        axes[0].add_patch(plt.Rectangle((x0 - 0.5, y0 - 0.5), size, size,
                                         fill=False, edgecolor="red", linewidth=2))
        axes[0].set_title(f"Image with kernel at (x={x}, y={y})")
        axes[0].axis("off")

        axes[1].imshow(kernel, cmap="RdBu_r", vmin=-kmax, vmax=kmax)
        # add a colorbar to the right of the kernel plot
        axes[1].figure.colorbar(axes[1].imshow(kernel, cmap="RdBu_r", vmin=kmin, vmax=kmax),
                                ax=axes[1], fraction=0.046, pad=0.04)
        
        axes[1].set_title(f"Kernel: {kernel_name}")
        axes[1].axis("off")

        axes[2].imshow(product, cmap="RdBu_r", vmin=-pmax, vmax=pmax)
        axes[2].figure.colorbar(axes[2].imshow(product, cmap="RdBu_r", vmin=pmin, vmax=pmax),
                                ax=axes[2], fraction=0.046, pad=0.04)   
        
        axes[2].set_title(f"patch × kernel,  sum = {result:.3f}")
        axes[2].axis("off")

        axes[3].imshow(full_convolution, cmap="gray")
        axes[3].scatter(x, y, s=100, color="yellow", marker="x")
        
        axes[3].set_title(f"Full convolution, \n({x}, {y}) = {pixel_val:.3f}")
        axes[3].axis("off")

        plt.tight_layout()
        plt.show()

    return interact(
        update,
        kernel_name=Dropdown(options=list(kernels.keys()),
                             value=default_kernel_name,
                             description="Kernel"),
        x=IntSlider(description="x", min=half, max=W - half - 1, value=W // 2),
        y=IntSlider(description="y", min=half, max=H - half - 1, value=H // 2),
    )

def get_pixel_values(Image, box=8):
    """
    Interactive pixel-patch viewer.

    Layout:
            X slider (horizontal, on top)
        ┌────────────────────┐ ┌───┐  ┌──────────────┐
        │                    │ │   │  │              │
        │       image        │ │ Y │  │ pixel table  │
        │                    │ │   │  │              │
        └────────────────────┘ └───┘  └──────────────┘

    The Y slider is rotated 180° so that 0 is at the top and H is at the
    bottom -- matching image-coordinate convention (row 0 = top row).
    """
    H, W = Image.shape[:2]
    half = box // 2

    # X slider: horizontal, above the image
    x_slider = IntSlider(
        value=W // 2, min=half, max=W - half - 1,
        continuous_update=True, orientation="horizontal",
        readout=True,
        layout=Layout(width="250px"),
    )
    x_label = Label(f"X: {x_slider.value}")
    x_slider.observe(lambda c: setattr(x_label, "value", f"X: {c['new']}"),
                     names="value")

    # Y slider: vertical, rotated 180° so the visual top corresponds to y=0.
    # The description and readout both rotate with the slider, so we hide
    # them and use a separate Label above the slider instead.
    y_slider = IntSlider(
        value=H // 2, min=half, max=H - half - 1,
        continuous_update=True, orientation="vertical",
        readout=False,
        layout=Layout(height="500px", transform="rotate(90deg)"),
    )
    y_label = Label(f"Y: {y_slider.value}")
    y_slider.observe(lambda c: setattr(y_label, "value", f"Y: {c['new']}"),
                     names="value")

    image_out = Output()
    table_out = Output()

    def redraw(*_):
        x, y = x_slider.value, y_slider.value
        i0, i1 = y - half, y + half
        j0, j1 = x - half, x + half
        y = H - 1 - y  # flip y to match image coordinates (row 0 = top row)
        # Image panel
        with image_out:
            image_out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(Image, cmap="gray")
            ax.add_patch(patches.Rectangle(
                (j0 - 0.5, i0 - 0.5), box, box,
                fill=False, edgecolor="yellow", linewidth=2,
            ))
            ax.set_title(f"Patch at (x={x}, y={y})")
            ax.axis("off")
            plt.show()
            plt.close(fig)

        # Pixel-value table
        with table_out:
            table_out.clear_output(wait=True)
            patch = Image[i0:i1, j0:j1]
            df = pd.DataFrame(
                patch,
                index=[f"y={i}" for i in range(i0, i1)],
                columns=[f"x={j}" for j in range(j0, j1)],
            )
            fmt = "{:d}" if np.issubdtype(patch.dtype, np.integer) else "{:.2f}"
            vmin = float(np.min(Image)); vmax = float(np.max(Image))
            styled = (
                df.style
                  .background_gradient(cmap="gray", vmin=vmin, vmax=vmax)
                  .format(fmt)
                  .set_properties(**{"text-align": "center",
                                     "font-size": "11px",
                                     "border": "1px solid #ddd",
                                     "padding": "4px"})
            )
            display(styled)

    x_slider.observe(redraw, names="value")
    y_slider.observe(redraw, names="value")

    # Compose layout
    x_panel = VBox([x_label, x_slider])                      # X label + slider on top
    y_panel = VBox([y_label, y_slider])                      # Y label + rotated slider
    image_panel = VBox([x_panel, HBox([image_out, y_panel])])
    ui = HBox([image_panel, table_out])

    redraw()
    return ui