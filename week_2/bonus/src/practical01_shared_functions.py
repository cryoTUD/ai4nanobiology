# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 16:23:05 2020

@author: abharadwaj1
"""

# imports
import mrcfile
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import resize


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
