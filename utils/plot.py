import numpy as np
import scipy as sp

def plot_line_and_ribbon(ax, x, y, color, alpha=0.25, **plt_kwargs):
    ax.plot(x, np.nanmean(y, 0), color=color, **plt_kwargs)
    ax.fill_between(x, np.nanmean(y, 0)-sp.stats.sem(y), np.nanmean(y, 0)+sp.stats.sem(y), color=color, alpha=alpha)