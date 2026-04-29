
# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ipywidgets import (
    IntSlider,
    fixed,
    interact_manual,
)
from IPython.display import display

def getbox(width, N=1000, index=500, high=1, low=0):
    """returns a box function of lenght N and box size a
    Note: signal[mid-halfbox] = high
          signal[mid+halfbox] = low
    """
    signal = np.zeros(N) + low
    halfbox = int(width / 2)
    signal[index - halfbox : index + halfbox + 1] = high
    return signal

# functions
def get_pixel_values(Image):
    return interact_manual(
        display_df,
        Image=fixed(Image),
        i=IntSlider(description="Shift Y", value=50, max=240, min=5),
        j=IntSlider(description="Shift X", value=50, max=240, min=5),
        box=fixed(8),
        continuous_update=True,
    )


def display_df(Image, i, j, box):
    df = pd.DataFrame(Image)
    df1 = df.iloc[
        int(i - box / 2) : int(i + box / 2),
        int(j - box / 2) : int(j + box / 2),
    ]
    # df1_styled = df1.style.background_gradient(cmap='viridis')
    rect = patches.Rectangle(
        (j - box / 2, i - box / 2), box, box, fill=False, color="yellow"
    )
    f = plt.figure(figsize=(8, 8))
    ax2 = f.add_subplot(122)
    ax2.imshow(Image, cmap="gray")
    ax2.add_patch(rect)
    # display df1 as a table
    display(df1)
    # df1.head()
