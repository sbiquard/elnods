#!/usr/bin/env python3

import argparse
import os
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import OptimizeResult

from models import Model


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="print more information",
    )
    parser.add_argument(
        "--thinfp", type=int, default=32, help="thin the focalplane by this much"
    )
    parser.add_argument("--real", type=int, default=0, help="realization number")
    parser.add_argument(
        "--outdir",
        type=str,
        default="elnod_out",
        help="output directory for plots and data files",
    )
    parser.add_argument(
        "--run-noisy-fit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run the noisy fit case",
    )
    parser.add_argument(
        "--run-wrong-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run the wrong model case",
    )
    parser.add_argument(
        "--run-pointing-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run the pointing error case",
    )
    parser.add_argument(
        "--perr",
        type=float,
        default=1 / 60,  # 1 arcminute
        help="[pointing error] typical elevation error in degrees",
    )
    parser.add_argument(
        "--squash-fp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="[pointing error] put all detectors at the same elevation",
    )
    parser.add_argument(
        "--same-offset",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="[pointing error] add the same elevation offset to all detectors",
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
    rng=None,
    realization: Optional[int] = None,
    tau=None,
    mean_g=None,
    eps=None,
):
    # if not provided, get a random number generator with optional seed
    if rng is None:
        rng = np.random.default_rng(realization)

    if not (tau is not None and mean_g is not None and eps is not None):
        # not all parameters provided, generate them
        ndet = x.shape[0]
        tau = rng.uniform(low=0.01, high=0.1)
        mean_g = rng.uniform(low=200, high=300)
        eps = rng.normal(loc=0, scale=0.1, size=ndet)

    # make sure the mean is zero
    eps -= np.mean(eps)

    # use the "true" model
    from models import ExpModel2

    model = ExpModel2()
    true_params = np.r_[tau, mean_g, eps]
    y = model.evaluate(true_params, x)
    if noise:
        scale = 0.01
        y += rng.normal(scale=scale, size=y.shape)
    return y, tau, mean_g, eps


def plot_resid(
    model: Model,
    dets: list[str],
    fit_result: OptimizeResult,
    x: np.ndarray,
    y: np.ndarray,
    title: Optional[str] = None,
    relative: bool = False,
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
    plot: bool = False,
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


def save_result(args, dets, base_name, rel_g_true, rel_g_fit, elev_bias=None):
    """Save the results to a .npz file"""
    data_out = os.path.join(args.outdir, "data")
    os.makedirs(data_out, exist_ok=True)
    filename = os.path.join(data_out, f"{base_name}_{len(dets)}_{args.real}.npz")
    if args.verbose:
        print(f"Saving results to {filename}")
    if elev_bias is not None:
        np.savez(filename, true=rel_g_true, estimate=rel_g_fit, elev_bias=elev_bias)
    else:
        np.savez(filename, true=rel_g_true, estimate=rel_g_fit)


def main(args):
    from models import ExpModel1, LinearModel1

    if not (args.run_noisy_fit or args.run_wrong_model or args.run_pointing_error):
        print("Nothing to do. Exiting.")
        return

    # get the data from TOAST
    ob, dets, times, elevs, x_toast, y_toast = prepare_data(args)

    # get a random number generator
    rng = np.random.default_rng(args.real)

    # simulate some fake data following our "true" model
    y_fake, tau_true, mean_g_true, eps_true = make_fake_data(
        x_toast, rng=rng, noise=False
    )
    rel_g_true = 1 + eps_true

    # noisy fit: fit a correct model to noisy TOAST data
    # --------------------------------------------------

    if args.run_noisy_fit:
        if args.verbose:
            print("----- Running 'noisy fit' case -----")

        # get the scrambled gains
        rel_g_toast = np.asarray(list(ob.scrambled_gains.values()))
        # ensure a central value of 1
        rel_g_toast /= np.mean(rel_g_toast)

        linear_model = LinearModel1()
        rel_g_fit_noisy = perform_fit(
            linear_model, x_toast, y_toast, dets, verbose=args.verbose
        )
        save_result(args, dets, "noisy_fit", rel_g_toast, rel_g_fit_noisy)

    # wrong model: fit a wrong model to fake data
    # -------------------------------------------

    if args.run_wrong_model:
        if args.verbose:
            print("\n----- Running 'wrong model' case -----")

        # fit a linear model
        linear_model = LinearModel1()
        rel_g_fit_wrong = perform_fit(
            linear_model, x_toast, y_fake, dets, verbose=args.verbose
        )
        save_result(args, dets, "wrong_model", rel_g_true, rel_g_fit_wrong)

    # pointing error
    # --------------

    if args.run_pointing_error:
        if args.verbose:
            print("\n----- Running 'pointing error' case -----")

        # add systematic errors to the elevation
        ndet = len(dets)
        if args.squash_fp:
            # put all detectors at the same elevation
            _elevs = np.tile(elevs[0], (ndet, 1))
            # we need to re-compute y_fake from the new elevations
            y_fake, _, _, _ = make_fake_data(
                1 / np.sin(_elevs),
                tau=tau_true,
                mean_g=mean_g_true,
                eps=eps_true,
                noise=False,
            )
        else:
            _elevs = elevs.copy()
        # add a constant offset to all detectors
        if args.same_offset:
            bias = np.deg2rad(rng.normal(loc=0, scale=args.perr))
        else:
            bias = np.deg2rad(rng.normal(loc=0, scale=args.perr, size=(ndet, 1)))
        elevs_biased = _elevs + bias
        x_biased = 1 / np.sin(elevs_biased)

        # fit the correct model but with biased x
        exp_model = ExpModel1()
        rel_g_fit_perror = perform_fit(
            exp_model, x_biased, y_fake, dets, verbose=args.verbose
        )
        save_result(
            args, dets, "pointing_error", rel_g_true, rel_g_fit_perror, elev_bias=bias
        )


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
