import matplotlib.pyplot as plt
import numpy as np
import pickle

def save_plot(fig, filename="shared_plot.pkl"):
    """Save matplotlib figure to a file"""
    with open(filename, "wb") as f:
        pickle.dump(fig, f)

def load_plot(filename="shared_plot.pkl"):
    """Load matplotlib figure from a file"""
    with open(filename, "rb") as f:
        fig = pickle.load(f)
    return fig