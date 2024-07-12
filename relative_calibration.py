# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import types

import numpy as np
import traitlets
from astropy import units as u
from scipy.optimize import Bounds, least_squares, minimize
from scipy.sparse import vstack, csr_matrix

from toast.qarray import to_iso_angles
from toast.observation import default_values as defaults
from toast.timing import Timer, function_timer
from toast.traits import Bool, Float, Int, Quantity, Unicode, trait_docs
from toast.utils import Environment, Logger
from toast.ops.operator import Operator


@trait_docs
class ElnodCalibration(Operator):
    """Perform relative calibration of detector gains using elnod intervals.

    A least squares fit to an elnod model is performed for every interval.
    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    det_data = Unicode(
        defaults.det_data,
        help="Observation detdata key containing the signal to use for calibration",
    )

    det_mask = Int(
        defaults.det_mask_invalid,
        help="Bit mask value for per-detector flagging",
    )

    bad_fit_mask = Int(
        defaults.det_mask_processing, help="Bit mask to raise for bad fits"
    )

    chi2 = Bool(
        True,
        help="Directly minimize sum(residual ** 2) instead of using a vector-valued fitting function",
    )

    @traitlets.validate("det_mask")
    def _check_det_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Det mask should be a positive integer")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        for ob in data.obs:
            dets = ob.select_local_detectors(selection=detectors)
            ndet = len(dets)

            # We will use the best fit parameters from each elnod as
            # the starting guess for the next elnod.
            params = None

            # Loop on the elnod intervals
            elnods = ob.view["elnod"]
            rel_gains = []
            for ielnod, elnod in enumerate(elnods):
                # Compute elevation angles for all detectors during the elnod
                quats = ob.detdata["quats_azel"][dets, elnod]
                elevs = self._el_from_quat(quats).reshape(ndet, -1)

                # Perform the fit on the data
                props = self._fit_elnod(
                    elevs, ob.detdata[self.det_data][dets, elnod], guess=params
                )

                if props["fit_result"].success:
                    # This was a good fit
                    params = props["fit_result"].x
                else:
                    params = np.r_[props["tau"], props["gains"]]
                    msg = (
                        f"Elnod fit failed in {ielnod}th elnod of observation {ob.name}"
                    )
                    log.warning(msg)
                    msg = f"  Best Result = {props['fit_result']}"
                    log.verbose(msg)
                    # TODO: still use the best result ?
                    # TODO: raise some flags ?

                # Accumulate measured gains (one per elnod interval)
                rel_gains.append({det: params[1 + i] for i, det in enumerate(dets)})

            # For now just store the result in the observation
            ob.rel_gains = rel_gains

            # TODO: use this to perform calibration

        return

    def _evaluate_model(self, elevs, tau, gains):
        """Evaluate the elnod model for multiple detectors.

        Given the input elevations, optical depth and relative gains, evaluate
        the elnods as

        elnods[det] = gains[det] * [ 1 - e^{-tau / sin(elevs[det])} ]

        Args:
            elevs (array):  The elevations in radians, of shape (ndet, nsamp)
            tau (float):  The optical depth (same for all detectors)
            gains (array):  The relative gains of the detectors

        Returns:
            (array):  The model elnods

        """
        # gains are broadcasted on the samples
        elnods = (-np.expm1(-tau / np.sin(elevs)).T * gains).T
        return elnods

    def _fit_fun(self, x, *args, **kwargs):
        """Evaluate the residual.

        For the given set of parameters, this evaluates the model elnods and computes the
        residual from the real data.

        Args:
            x (array):  The current model parameters
            kwargs:  The fixed information passed in through the least squares solver.

        Returns:
            (array):  The array of residuals

        """
        elevs = kwargs["elevs"]
        data = kwargs["data"]
        tau = x[0]
        gains = x[1:]
        current = self._evaluate_model(elevs, tau, gains)
        resid = current - data
        # flatten the residual
        return resid.ravel()

    def _fit_jac(self, x, *args, **kwargs):
        """Evaluate the partial derivatives of model.

        This returns the Jacobian containing the partial derivatives of the model
        with respect to the fit parameters.

        Args:
            x (array):  The current model parameters
            kwargs:  The fixed information passed in through the least squares solver.

        Returns:
            (CSR matrix):  The Jacobian (as a sparse matrix in CSR format)

        """
        elevs = kwargs["elevs"]
        tau = x[0]
        gains = x[1:]
        ndet, nsamp = elevs.shape

        # Build the sparse Jacobian
        jacs = []

        for i in range(ndet):
            # TODO: this seems a bit wasteful, can it be done more efficiently ?
            jac = np.zeros((nsamp, x.size))

            # 1st column = partial derivative wrt to tau
            sin_el = np.sin(elevs[i])
            jac[:, 0] = gains[i] * np.exp(-tau / sin_el) / sin_el

            # column i+1 = partial derivative wrt to g_i
            jac[:, i + 1] = -np.expm1(-tau / sin_el)

            # convert and store as CSR matrix
            jacs.append(csr_matrix(jac))

        # stack all the blocks as a new CSR matrix
        J = vstack(jacs, format="csr")
        return J

    def _chi2_fun_jac(self, x, *args):
        """Compute objective function and its gradient at the same time"""
        # unpack arguments
        elevs, data = args
        tau = x[0]
        gains = x[1:]

        # elnod model for current vector
        sin_elev = np.sin(elevs)
        emission = -np.expm1(-tau / sin_elev)
        current = (emission.T * gains).T

        # objective
        resid = current - data
        obj = np.square(resid).ravel().sum()

        # gradient
        grad = np.empty_like(x)

        # partial derivative wrt g's
        partial_g = emission
        grad[1:] = 2 * np.sum(resid * partial_g, axis=-1)

        # partial derivative wrt tau
        partial_tau = (np.exp(-tau / sin_elev).T * gains).T / sin_elev
        grad[0] = 2 * (resid * partial_tau).ravel().sum()

        return obj, grad

    def _chi2_hess(self, x, *args):
        """Compute the Hessian of the objective function"""
        # unpack arguments
        elevs, data = args
        tau = x[0]
        gains = x[1:]

        # intermediate quantities
        sin_elev = np.sin(elevs)
        tau_el = tau / sin_elev
        emission = -np.expm1(-tau_el)
        current = (emission.T * gains).T
        resid = current - data

        # initialize the Hessian with zeros
        hess = np.zeros((x.size, x.size))

        # tau x tau
        partial_tau = (np.exp(-tau_el).T * gains).T / sin_elev
        hess[0, 0] = 2 * np.sum(np.square(partial_tau) - resid * partial_tau / sin_elev)

        # g x g
        # only the diagonal is non-zero
        partial_g = emission
        gg_diag = np.einsum("ii->i", hess)  # view of the diagonal
        gg_diag[1:] = 2 * np.einsum("ij,ij->i", partial_g, partial_g)

        # tau x g
        hess[0, 1:] = 2 * np.sum(
            (current + resid) * np.exp(-tau_el) / sin_elev, axis=-1
        )
        hess[1:, 0] = hess[0, 1:]

        return hess

    def _get_err_ret(self, ndet):
        # Internal function to build a fake return result
        # when the fitting fails for some reason.
        eret = dict()
        eret["fit_result"] = types.SimpleNamespace()
        eret["fit_result"].success = False
        eret["tau"] = 0.0
        eret["gains"] = np.ones(ndet)
        return eret

    @staticmethod
    def _el_from_quat(azel_quat):
        # Convert Az/El quaternion of the detector back into angles
        theta, _, _ = to_iso_angles(azel_quat)
        el = np.pi / 2 - theta
        return el

    def _fit_elnod(self, elevs, data, guess=None):
        """Perform a fit to model elnod parameters.

        Args:
            elevs (Quantity):  The elevation angles for all detectors
            data (Quantity):  The measured elnod signal
            guess (array):  Optional starting point guess

        Returns:
            (dict):  Dictionary of fit parameters

        """
        log = Logger.get()
        ret = dict()

        # TODO: any reason for skipping part of the interval ?
        # TODO: set bounds ?

        ndet = elevs.shape[0]
        if guess is None:
            x_0 = np.r_[0.01, np.ones(ndet)]
        else:
            x_0 = guess

        # print(f"FIT:  starting guess = {x_0}")

        try:
            if self.chi2:
                result = minimize(
                    self._chi2_fun_jac,
                    x_0,
                    args=(elevs, data),
                    jac=True,  # function returns objective and gradient
                    hess=self._chi2_hess,
                    method="Newton-CG",
                    options={
                        "xtol": 1e-10,
                        "maxiter": 500 * x_0.size,
                    },
                )
            else:
                result = least_squares(
                    self._fit_fun,
                    x_0,
                    jac=self._fit_jac,
                    max_nfev=200 * x_0.size,  # twice the scipy default
                    x_scale="jac",  # TODO: quantify improvement from this
                    verbose=0,
                    kwargs={
                        "elevs": elevs,
                        "data": data,
                    },
                )
        except Exception:
            log.verbose("PSD fit raised exception, skipping")
            ret = self._get_err_ret(ndet)
            return ret

        # print(f"FIT result = {result}")

        ret["fit_result"] = result
        if result.success:
            ret["tau"] = result.x[0]
            ret["gains"] = result.x[1:]
        else:
            ret["tau"] = 0.0
            ret["gains"] = np.ones(ndet)

        return ret

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": list(),
            "shared": list(),
            "detdata": [self.det_data, "quats_azel"],
            "intervals": ["elnod"],
        }
        return req

    def _provides(self):
        return dict()
