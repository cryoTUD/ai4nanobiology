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
