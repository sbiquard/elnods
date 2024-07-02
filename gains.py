#!/usr/bin/env python3

import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.optimize import OptimizeResult
from typing import Optional

from models import Model


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="print more information"
    )
    parser.add_argument(
        "--thinfp",
        type=int,
        default=64,
        help="thin the focalplane by this much"
    )
    parser.add_argument(
        "--real", type=int, default=0, help="realization number"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="elnod_out",
        help="output directory for plots and data files"
    )
    return parser


def prepare_data(args):
    import toast
    import utils

    # simulate TOAST data with an elnod at the start of the observation
    env = toast.Environment.get()
    comm, procs, rank = toast.get_world()
    moves = [0.0, +2.0, -2.0, 0.0]
    data = utils.simulate_data(
        moves,
        comm,
        noise=True,
        elevation_noise=True,
        atm_fluctuations=True,
        scramble_gains=True,
        thinfp=args.thinfp,
        realization=args.real,
    )

    # get what we need from the observation
    ob = data.obs[0]
    dets = ob.local_detectors
    ndet = len(dets)
    elnod = ob.view["elnod"][0]
    times = np.arange(elnod.start, elnod.stop)
    quats = ob.detdata["quats_azel"][dets, elnod]
    elevs = utils.get_el_from_quat(quats).reshape(ndet, -1)
    x_toast = 1 / np.sin(elevs)
    y_toast = ob.detdata["signal"][dets, elnod]

    return ob, dets, times, elevs, x_toast, y_toast


def make_fake_data(
    x: np.ndarray,
    noise: bool = False,
    realization: Optional[int] = None,
):
    # get a random number generator with optional seed
    rng = np.random.default_rng(realization)

    ndet = x.shape[0]
    tau = rng.uniform(low=0.01, high=0.1)
    mean_g = rng.uniform(low=200, high=300)
    eps = rng.normal(loc=0, scale=0.1, size=ndet)
    # make sure the mean is zero
    eps -= np.mean(eps)
    rel_g = 1 + eps

    # use the "true" model
    from models import ExpModel2

    model = ExpModel2()
    true_params = np.r_[tau, mean_g, eps]
    y = model.evaluate(true_params, x)
    if noise:
        scale = 0.01
        y += rng.normal(scale=scale, size=y.shape)
    return y, tau, mean_g, rel_g


def plot_resid(
    model: Model,
    dets: list[str],
    fit_result: OptimizeResult,
    x: np.ndarray,
    y: np.ndarray,
    title: Optional[str] = None,
    relative: bool = False
) -> None:
    # Plot fit results
    plt.figure(figsize=(10, 4))
    fit_params = fit_result.x
    resid = y - model.evaluate(fit_params, x)
    cmap = mpl.colormaps["Paired"]
    ndet = len(dets)
    t = np.arange(x.shape[1])
    for i, det in enumerate(dets):
        r = resid[i]
        if relative:
            r /= y[i]
        color, alpha = (cmap(i), 1.0) if ndet <= 12 else ("k", 0.5)
        plt.plot(t, r, color=color, alpha=alpha, label=det)
    plt.title("Residuals" if title is None else title)
    if ndet <= 12:
        plt.legend(ncol=2)
    plt.axhline(0, color="k", linestyle="dotted")
    plt.show()


def perform_fit(
    model: Model,
    x: np.ndarray,
    y: np.ndarray,
    dets: list[str],
    verbose: bool = True,
    plot: bool = False
) -> np.ndarray:
    # Perform a fit and return the relative gains
    from time import perf_counter

    t0 = perf_counter()
    fit_result = model.fit(x, y, disp=args.verbose)
    dt = int((perf_counter() - t0) * 1e3)  # milliseconds

    if verbose:
        ndet = len(dets)
        print(f"Model fitting took {dt} ms")
        print(f"({dt / ndet / fit_result.nit:.3} ms / detector / iteration)")

    if plot:
        plot_resid(model, dets, fit_result, x, y)

    # return the fitted relative gains
    rel_g_fit = model.rel_gains(fit_result.x)
    return rel_g_fit


def save_result(args, dets, rel_g_true, rel_g_fit, base_name):
    """Save the results to a .npz file"""
    os.makedirs(args.outdir, exist_ok=True)
    filename = os.path.join(
        args.outdir, f"{base_name}_{len(dets)}_{args.real}.npz"
    )
    if args.verbose:
        print(f"Saving results to {filename}")
    np.savez(filename, true=rel_g_true, estimate=rel_g_fit)


def main(args):
    from models import LinearModel1

    # get the data from TOAST
    ob, dets, times, elevs, x_toast, y_toast = prepare_data(args)

    # noisy fit: fit a correct model to noisy TOAST data
    # --------------------------------------------------

    if args.verbose:
        print("----- Running 'noisy fit' case -----")

    # get the scrambled gains
    rel_g_toast = np.asarray(list(ob.scrambled_gains.values()))
    # ensure a central value of 1
    rel_g_toast /= np.mean(rel_g_toast)

    model = LinearModel1()
    rel_g_fit_noisy = perform_fit(model, x_toast, y_toast, dets, verbose=args.verbose)
    save_result(args, dets, rel_g_toast, rel_g_fit_noisy, "noisy_fit")

    # wrong model: fit a wrong model to fake data
    # -------------------------------------------

    if args.verbose:
        print("\n----- Running 'wrong model' case -----")

    # simulate some fake data following our "true" model
    y_fake, tau_true, mean_g_true, rel_g_true = make_fake_data(
        x_toast, noise=False, realization=args.real
    )

    # fit a linear model
    rel_g_fit_wrong = perform_fit(model, x_toast, y_fake, dets, verbose=args.verbose)
    save_result(args, dets, rel_g_toast, rel_g_fit_wrong, "wrong_model")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
