import matplotlib.pyplot as plt
import numpy as np
import pickle

def create_plot(x=None, y=None):
    """Create a simple plot (default: sine curve)"""
    if x is None or y is None:
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    return fig, ax

def save_plot(fig, filename="shared_plot.pkl"):
    """Save matplotlib figure to a file"""
    with open(filename, "wb") as f:
        pickle.dump(fig, f)

def load_plot(filename="shared_plot.pkl"):
    """Load matplotlib figure from a file"""
    with open(filename, "rb") as f:
        fig = pickle.load(f)
    return fig