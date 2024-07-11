#!/usr/bin/env python3

import argparse
import os
import re

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


FLAVORS = ["noisy_fit", "wrong_model", "pointing_error"]


@dataclass
class Info:
    ndet_max: int = 0
    nreal: int = 0


def explore(data_path):
    infos = {flavor: Info() for flavor in FLAVORS}
    p = re.compile(r"(\D*)_(\d+)_(\d+)\.npz")
    for child in data_path.iterdir():
        fname = os.path.basename(child)
        m = p.match(fname)
        flavor, ndet, real = m.groups()
        info = infos[flavor]
        info.ndet_max = max(info.ndet_max, int(ndet))
        info.nreal += 1
    return infos


def load(data_path):
    infos = explore(data_path)
    data = {
        flavor: np.empty((2, info.nreal, info.ndet_max), dtype=np.float64)
        for flavor, info in infos.items()
    }
    for flavor in data:
        df = data[flavor]
        ndet_max = df.shape[-1]
        for i, file in enumerate(data_path.glob(flavor + "*.npz")):
            content = np.load(file)
            # put NaNs if missing detectors
            pad_shape = (0, ndet_max - content["true"].size)
            df[0, i] = np.pad(content["true"], pad_shape, constant_values=np.nan)
            df[1, i] = np.pad(content["estimate"], pad_shape, constant_values=np.nan)
    return data


def make_correl_plots(data, plot_path):
    nf = len(data)
    fig, axs = plt.subplots(1, nf, figsize=(4 * nf, 4), sharex="row", sharey="row")

    # helper function
    def func(ax, values):
        ax.scatter(values[0], values[1], marker=".")
        xmin, xmax = ax.get_xlim()
        ax.plot([xmin, xmax], [xmin, xmax], color="black", linestyle="--")
        ax.set_xlabel("True value")
        ax.set_ylabel("Recovered value")

    for i, (flavor, df) in enumerate(data.items()):
        ax = axs[i]
        func(ax, df)
        ax.set_title(flavor)
    for ax in axs:
        ax.label_outer()
    fig.tight_layout()
    plt.savefig(plot_path / "correl")


def make_histograms(data, plot_path, distribution, distribution_pairs):
    nf = len(data)
    fig, axs = plt.subplots(1, nf, figsize=(4 * nf, 4))

    # helper function
    def func(ax, values, dist: Optional[str] = None):
        # plot the histogram
        ax.hist(values, density=True, bins="auto")
        ax.set_xlabel("Error")
        ax.set_ylabel("Density")

        # compute statistics
        mu = np.nanmean(values)
        std = np.nanstd(values)
        textstr = "\n".join((r"$\mu=%.2e$" % (mu,), r"$\sigma=%.2e$" % (std,)))

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

        # place a text box in upper left in axes coords
        ax.text(
            0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=props,
        )

        if dist is not None:
            import scipy.stats

            # fit a distribution on top
            stat = getattr(scipy.stats, dist)
            *shape, loc, scale = stat.fit(values)

            xmin, xmax = ax.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = stat.pdf(x, *shape, loc=loc, scale=scale)
            ax.plot(
                x,
                p,
                "r--",
                linewidth=1,
                label=f"{dist} fit",
            )
            ax.legend(loc="upper right")

    for i, (flavor, df) in enumerate(data.items()):
        ax = axs[i]
        diff = df[1] - df[0]
        func(ax, diff.ravel(), dist=distribution)
        ax.set_title(flavor)
    # for ax in axs:
    #     ax.label_outer()
    fig.tight_layout()
    plt.savefig(plot_path / "histo")

    # now for the detectors pairs
    fig, axs = plt.subplots(1, nf, figsize=(4 * nf, 4))
    for i, (flavor, df) in enumerate(data.items()):
        ax = axs[i]
        diff = df[1] - df[0]
        diff_pairs = diff[:, ::2] - diff[:, 1::2]
        func(ax, diff_pairs.ravel(), dist=distribution_pairs)
        ax.set_title(flavor)
    # for ax in axs:
    #     ax.label_outer()
    fig.tight_layout()
    plt.savefig(plot_path / "histo_pairs")


def main(args):
    # set the context using seaborn
    sns.set_context(args.context)

    # load the data
    data_path = Path(args.outdir) / "data"
    data = load(data_path)

    # make some plots
    plot_path = Path(args.outdir) / "plots"
    plot_path.mkdir(exist_ok=True)

    if args.make_correl_plots:
        make_correl_plots(data, plot_path)

    if args.make_histograms:
        make_histograms(data, plot_path, args.dist, args.dist_pairs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default="out",
        help="directory where to look for outputs",
    )
    parser.add_argument(
        "--context",
        choices=["paper", "notebook", "talk", "poster"],
        default="paper",
        help="context to scale the plot elements (cf. seaborn documentation)",
    )
    parser.add_argument(
        "--make-correl-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="produce correlation plots between true and recovered values",
    )
    parser.add_argument(
        "--make-histograms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="produce histograms of errors",
    )
    parser.add_argument(
        "--dist",
        choices=["norm", "cauchy", "skewnorm", "skewcauchy"],
        default="skewnorm",
        help="type of probability distribution to use for histogram fits",
    )
    parser.add_argument(
        "--dist-pairs",
        choices=["norm", "cauchy", "skewnorm", "skewcauchy"],
        default="cauchy",
        help="type of probability distribution to use for histogram fits of pairs",
    )
    args = parser.parse_args()
    main(args)
