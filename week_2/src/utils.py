"""
Utility functions for Module 1: MNIST classification with FC networks.
"""

import io
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


# ============================================================
# Data loading
# ============================================================

def get_mnist_loaders(batch_size=128, val_fraction=0.1, train_subset_size=None,
                     data_root="./data", seed=0):
    """
    Returns (train_loader, val_loader, test_loader).

    - MNIST has 60,000 train + 10,000 test by default.
    - We carve `val_fraction` of the training set into a validation set.
    - If `train_subset_size` is given, we use only that many training examples
      (useful for inducing overfitting deliberately).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean/std
    ])

    train_full = datasets.MNIST(root=data_root, train=True,  download=True, transform=transform)
    test_set   = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    # Optional shrink BEFORE the val split, so val stays a clean held-out set
    if train_subset_size is not None:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(train_full), train_subset_size, replace=False).tolist()
        train_full = Subset(train_full, idx)

    n_val = int(len(train_full) * val_fraction)
    n_train = len(train_full) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(train_full, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


# ============================================================
# Visualization
# ============================================================

def show_examples(loader, n=12, title="MNIST examples"):
    """Display n example images with their class labels."""
    x, y = next(iter(loader))
    n = min(n, x.size(0))
    fig, axes = plt.subplots(2, n // 2, figsize=(n * 0.9, 2.5))
    for i, ax in enumerate(axes.flat):
        img = x[i].squeeze().numpy()
        ax.imshow(img, cmap="gray")
        ax.set_title(f"label: {int(y[i])}", fontsize=9)
        ax.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_loss_curves(train_losses, val_losses, train_accs=None, val_accs=None):
    """
    Plot training/validation curves side by side.

    Pass train_accs/val_accs as well to also see the accuracy curves.
    Look for: training loss -> 0 while val loss flattens or rises = overfitting.
    """
    has_acc = train_accs is not None and val_accs is not None
    n_panels = 2 if has_acc else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, label="train", marker="o", markersize=3)
    axes[0].plot(epochs, val_losses,   label="val",   marker="s", markersize=3)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Loss curves")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    if has_acc:
        axes[1].plot(epochs, train_accs, label="train", marker="o", markersize=3)
        axes[1].plot(epochs, val_accs,   label="val",   marker="s", markersize=3)
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("accuracy")
        axes[1].set_title("Accuracy curves")
        axes[1].grid(alpha=0.3)
        axes[1].legend()

    plt.tight_layout()
    plt.show()


def show_predictions(model, loader, device, n=12, only_wrong=False):
    """Show a grid of test images with the model's prediction and true label."""
    model.eval()
    images, true_labels, pred_labels = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = model(x).argmax(dim=1).cpu()
            for i in range(x.size(0)):
                if only_wrong and preds[i].item() == y[i].item():
                    continue
                images.append(x[i].cpu()); true_labels.append(int(y[i])); pred_labels.append(int(preds[i]))
                if len(images) >= n:
                    break
            if len(images) >= n:
                break

    if not images:
        print("No examples to show (possibly all predictions were correct?).")
        return

    n = len(images)
    cols = min(6, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.7))
    axes = np.atleast_2d(axes)
    for i in range(rows * cols):
        ax = axes.flat[i]
        if i < n:
            img = images[i].squeeze().numpy()
            ax.imshow(img, cmap="gray")
            correct = (true_labels[i] == pred_labels[i])
            color = "green" if correct else "red"
            ax.set_title(f"pred={pred_labels[i]}\ntrue={true_labels[i]}", fontsize=8, color=color)
        ax.axis("off")
    plt.suptitle("Wrong predictions" if only_wrong else "Predictions")
    plt.tight_layout()
    plt.show()


# ============================================================
# Hand-drawn digit handling
# ============================================================

def preprocess_drawn_digit(image_array_or_path, invert=True):
    """
    Take a hand-drawn digit (numpy array or path to a PNG) and convert it to
    the same format MNIST uses: 1x28x28, normalized.

    MNIST conventions:
      - white digit on black background
      - centered, ~20px tall, anti-aliased
      - normalized with mean 0.1307, std 0.3081

    Most paint programs save as black on white -> we invert by default.
    """
    from PIL import Image
    import numpy as np

    if isinstance(image_array_or_path, str):
        img = Image.open(image_array_or_path).convert("L")
    elif isinstance(image_array_or_path, np.ndarray):
        if image_array_or_path.ndim == 3:
            image_array_or_path = image_array_or_path.mean(axis=-1)
        img = Image.fromarray(image_array_or_path.astype(np.uint8))
    else:
        img = image_array_or_path  # assume PIL Image already

    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    if invert:
        arr = 1.0 - arr

    # MNIST normalization
    arr = (arr - 0.1307) / 0.3081
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1, 1, 28, 28]
    return tensor


def predict_drawn_digit(model, image_input, device):
    """Run a hand-drawn digit through the trained model. Returns (pred, probs)."""
    tensor = preprocess_drawn_digit(image_input).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
        pred = int(probs.argmax())
    return pred, probs


def show_drawn_digit_prediction(image_input, pred, probs):
    """Visualize the drawn digit and the model's class probabilities."""
    tensor = preprocess_drawn_digit(image_input)
    img = tensor.squeeze().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title(f"Predicted: {pred}")
    axes[0].axis("off")

    axes[1].bar(range(10), probs)
    axes[1].set_xticks(range(10))
    axes[1].set_xlabel("class")
    axes[1].set_ylabel("probability")
    axes[1].set_title("Class probabilities")
    axes[1].grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()


# ============================================================
# ipycanvas drawing widget
# ============================================================

def make_drawing_canvas(size=200):
    """
    Return an ipycanvas drawing widget plus a function to extract the
    current drawing as a numpy array suitable for preprocess_drawn_digit.

    Usage:
        canvas, get_image = make_drawing_canvas()
        display(canvas)
        # ... user draws ...
        arr = get_image()
        pred, probs = predict_drawn_digit(model, arr, device)

    Requires: pip install ipycanvas
    Falls back gracefully with a clear error if ipycanvas isn't installed.
    """
    try:
        from ipycanvas import Canvas
        from ipywidgets import Button, HBox, VBox, Output
    except ImportError:
        raise ImportError(
            "ipycanvas is not installed. Run:  pip install ipycanvas\n"
            "If that doesn't work in your environment, draw a digit in any "
            "paint program, save it as PNG, and use preprocess_drawn_digit('path.png')."
        )

    canvas = Canvas(width=size, height=size, sync_image_data=True)
    canvas.fill_style = "white"
    canvas.fill_rect(0, 0, size, size)
    canvas.stroke_style = "black"
    canvas.line_width = 12  # thick strokes -> closer to MNIST after downsize

    drawing = {"is_drawing": False, "last": None}

    def on_mouse_down(x, y):
        drawing["is_drawing"] = True
        drawing["last"] = (x, y)

    def on_mouse_move(x, y):
        if drawing["is_drawing"] and drawing["last"] is not None:
            canvas.stroke_line(drawing["last"][0], drawing["last"][1], x, y)
            drawing["last"] = (x, y)

    def on_mouse_up(x, y):
        drawing["is_drawing"] = False
        drawing["last"] = None

    canvas.on_mouse_down(on_mouse_down)
    canvas.on_mouse_move(on_mouse_move)
    canvas.on_mouse_up(on_mouse_up)

    def get_image():
        """Return the current canvas as a 2D numpy array (grayscale 0-255)."""
        arr = canvas.get_image_data()  # RGBA, shape [size, size, 4]
        gray = arr[:, :, :3].mean(axis=-1)  # to grayscale
        return gray  # caller can pass to preprocess_drawn_digit

    def clear():
        canvas.fill_style = "white"
        canvas.fill_rect(0, 0, size, size)

    return canvas, get_image, clear
