# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 16:23:05 2020

@author: abharadwaj1
"""

# imports
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import (
    interact,
    IntSlider,
    FloatSlider,
)


# functions
def play_with_wave():
    def plotwave(Amplitude, Frequency, Phase=0):
        times = np.linspace(0, 5, 1000)
        f = Amplitude * np.cos(2 * np.pi * Frequency * times - Phase)
        plt.plot(times, f)
        plt.xlabel("X axis ")
        plt.ylabel("Function, $f(x)$")
        plt.ylim([-5, 5])
        return

    return interact(
        plotwave,
        Amplitude=IntSlider(description="Amplitude", value=2, min=-4, max=4),
        Frequency=IntSlider(description="Freq ", value=4, min=1, max=8),
        Phase=FloatSlider(
            description="Phase (rad)",
            value=0.0,
            min=-2 * np.pi,
            max=2 * np.pi,
            step=0.5,
        ),
        continuous_update=False,
    )
