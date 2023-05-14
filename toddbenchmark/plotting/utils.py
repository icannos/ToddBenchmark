import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def perf_vs_rejection(df, perf_metric, score, sign=1):

    rejection_rate = np.arange(0, 1, 0.01)

    # Get average performance when rejecting a certain percentage of the data

    perf = []
    for rejection in rejection_rate:
        # Get the threshold for the rejection rate
        threshold = df[score].quantile(rejection)
        # Get the average performance for the threshold
        perf.append(df[df[score] * sign > threshold][perf_metric].mean())

    return rejection_rate, perf


def plot_perf_vs_rejection(df, perf_metric, score, sign=1):

    thresholds, perf = perf_vs_rejection(df, perf_metric, score, sign)

    plt.plot(thresholds, perf)
    plt.xlabel(f"Rejection rate for {score}")
    plt.ylabel(f"Average {perf_metric}")
    plt.show()
