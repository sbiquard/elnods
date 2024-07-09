from abc import ABC, abstractmethod
from typing import ClassVar, Optional

import numpy as np
import scipy.sparse as sps
from scipy.optimize import OptimizeResult, minimize

from utils import div_mean

__all__ = [
    "Model",
    "LinearModel1",
    "LinearModel2",
    "ExpModel1",
    "ExpModel2",
]


class Model(ABC):
    minimizer: ClassVar[str]
    description: ClassVar[str]
    plot_color: ClassVar[str]

    @staticmethod
    @abstractmethod
    def param_labels(ndet: int) -> list[str]:
        """Tick labels of paramters for use in plots"""

    def chi2_fun(self, params, x, y) -> float:
        """Compute the chi2 of the model, i.e. sum((y - model(x))**2)"""
        ypred = self.evaluate(params, x)
        return np.sum((y - ypred) ** 2)

    @abstractmethod
    def evaluate(self, params, x) -> np.ndarray:
        """Evaluate the model at x with parameters params"""

    @abstractmethod
    def _extract_params(self, params):
        """Build the tuple of parameters from the full parameter vector"""

    @abstractmethod
    def rel_gains(self, params) -> np.ndarray:
        """Relative gains from model parameters"""

    # minimization methods

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        init_params: Optional[np.ndarray] = None,
        disp: bool = False,
        tol: Optional[float] = None,
        maxiter_factor: Optional[int] = None,
        use_hess: bool = True,
        sparse_hessian: bool = True,
    ) -> OptimizeResult:
        """Perform the fit by minimizing the chi2 value"""
        if init_params is None:
            init_params = self._default_init_params(x.shape[0])

        kwargs = self._minimize_kwargs(
            x, y, init_params, disp, tol, maxiter_factor, use_hess, sparse_hessian
        )

        # perform the minimization
        res = minimize(self.chi2_fun_jac, init_params, **kwargs)
        return res

    @abstractmethod
    def _default_init_params(self, ndet: int) -> np.ndarray:
        """Default initial parameters for the minimization"""

    def _minimize_kwargs(
        self, x, y, init_params, disp, tol, maxiter_factor, use_hess, sparse_hessian
    ):
        """Get kwargs for the minimization function"""
        kwargs = {
            "args": (x, y),
            "jac": True,
            "method": self.minimizer,
            "tol": tol,
            "options": {"disp": disp},
        }

        if maxiter_factor is not None:
            kwargs["options"]["maxiter"] = maxiter_factor * init_params.size

        if use_hess:
            kwargs["hess"] = self.hess_sp if sparse_hessian else self.hess

        return kwargs

    # jacobian and hessian methods
    # specific to the model, must be implemented by the subclass

    @abstractmethod
    def chi2_fun_jac(self, params, x, y) -> tuple[float, np.ndarray]:
        """Compute the chi2 and its jacobian w.r.t. params"""

    @abstractmethod
    def hess(self, params, x, y) -> np.ndarray:
        """Compute the hessian of the model at x with parameters params"""

    @abstractmethod
    def hess_sp(self, params, x, y) -> sps.csr_matrix:
        """Compute the hessian of the model at x as a sparse CSR matrix"""


class LinearModel1(Model):
    minimizer = "Newton-CG"
    description = "linear $\\chi^2(g_i)$"
    plot_color = "tab:blue"

    @staticmethod
    def param_labels(ndet: int) -> list[str]:
        return ["$g_{" + str(_) + "}$" for _ in range(ndet)]

    def evaluate(self, params, x) -> np.ndarray:
        gains = self._extract_params(params)
        return (x.T * gains).T

    def _extract_params(self, params):
        # params are just the gains
        return params

    def rel_gains(self, params) -> np.ndarray:
        gains = self._extract_params(params)
        return div_mean(gains)

    def _default_init_params(self, ndet) -> np.ndarray:
        return np.ones(ndet)

    def chi2_fun_jac(self, params, x, y) -> tuple[float, np.ndarray]:
        ypred = self.evaluate(params, x)
        resid = y - ypred
        chi2 = np.sum(resid**2)
        grad = -2 * np.sum(resid * x, axis=-1)
        return chi2, grad

    def hess(self, params, x, y) -> np.ndarray:
        return 2 * np.diag(np.einsum("ij,ij->i", x, x))

    def hess_sp(self, params, x, y) -> sps.csr_matrix:
        m = params.size
        # indptr = np.arange(m + 1)
        # indices = np.arange(m)
        # values = np.einsum("ij,ij->i", x, x)
        # return csr_matrix((2 * values, indices, indptr), shape=(m, m))
        return sps.diags(2 * np.einsum("ij,ij->i", x, x), shape=(m, m), format="csr")  # type: ignore


class LinearModel2(Model):
    minimizer = "trust-constr"
    description = "linear $\\chi^2(g_0,\\epsilon_i)$"
    plot_color = "tab:orange"

    @staticmethod
    def param_labels(ndet: int) -> list[str]:
        return ["$g_0$"] + ["$\\epsilon_{" + str(_) + "}$" for _ in range(ndet)]

    def evaluate(self, params, x) -> np.ndarray:
        mean_g, rel_g = self._extract_params(params)
        gains = mean_g * rel_g
        return (x.T * gains).T

    def _extract_params(self, params):
        mean_g = params[0]
        rel_g = 1 + params[1:]
        return mean_g, rel_g

    def rel_gains(self, params) -> np.ndarray:
        _, rel_g = self._extract_params(params)
        return rel_g

    def _default_init_params(self, ndet) -> np.ndarray:
        return np.r_[1, np.zeros(ndet)]

    def _minimize_kwargs(
        self, x, y, init_params, disp, tol, maxiter_factor, use_hess, sparse_hessian
    ):
        # get the "standard" kwargs from the parent class
        kwargs = super()._minimize_kwargs(
            x, y, init_params, disp, tol, maxiter_factor, use_hess, sparse_hessian
        )

        # add a constraint on the epsilons
        from scipy.optimize import LinearConstraint

        ndet = x.shape[0]
        kwargs["constraints"] = LinearConstraint(
            np.r_[0, np.ones(ndet)].reshape(1, -1), lb=0, ub=0
        )
        return kwargs

    def chi2_fun_jac(self, params, x, y) -> tuple[float, np.ndarray]:
        ypred = self.evaluate(params, x)
        resid = y - ypred
        chi2 = np.sum(resid**2)

        # gradient
        grad = np.empty_like(params)
        mean_g, rel_g = self._extract_params(params)
        grad[0] = -2 * np.sum(resid * ypred) / mean_g
        grad[1:] = -2 * np.sum(resid * ypred, axis=-1) / rel_g

        return chi2, grad

    def hess(self, params, x, y) -> np.ndarray:
        mean_g, rel_g = self._extract_params(params)
        hess = np.zeros((params.size, params.size))
        h_diag = np.einsum("ii->i", hess)  # view of the diagonal

        # c x c
        x_square = x**2
        hess[0, 0] = (x_square.T * rel_g**2).T.sum()

        # c x eps
        hess[0, 1:] = np.sum((x_square.T * rel_g).T, axis=-1) * mean_g
        hess[1:, 0] = hess[0, 1:]

        # eps x eps
        h_diag[1:] = np.einsum("ij,ij->i", x, x) * mean_g**2

        return 2 * hess

    def hess_sp(self, params, x, y) -> sps.csr_matrix:
        mean_g, rel_g = self._extract_params(params)

        # c x c
        x_square = x**2
        c_x_c = (x_square.T * rel_g**2).T.sum()

        # c x eps
        c_x_eps = np.sum((x_square.T * rel_g).T, axis=-1) * mean_g

        # eps x eps
        eps_x_eps = np.einsum("ij,ij->i", x, x) * mean_g**2

        # CSR format
        m = params.size
        nnz = 3 * m - 2

        # 1st row is full, others have 2 nonzeros each
        indptr = np.r_[0, np.arange(m, nnz + 1, 2)]

        indices = np.empty(nnz)
        indices[:m] = np.arange(m)  # 1st line
        indices[m::2] = 0  # 1st column
        indices[m + 1 :: 2] = np.arange(1, m)  # diagonal

        values = np.empty(nnz)
        values[0] = c_x_c
        values[1:m] = c_x_eps
        values[m::2] = c_x_eps
        values[m + 1 :: 2] = eps_x_eps

        return sps.csr_matrix((2 * values, indices, indptr), shape=(m, m))


class ExpModel1(Model):
    minimizer = "Newton-CG"
    descrition = "exp $\\chi^2(\\tau,g_i)$"
    plot_color = "tab:green"

    @staticmethod
    def param_labels(ndet: int) -> list[str]:
        return ["$\\tau$"] + ["$g_{" + str(x) + "}$" for x in range(ndet)]

    def evaluate(self, params, x) -> np.ndarray:
        tau, gains = self._extract_params(params)
        emission = -np.expm1(-tau * x)
        return (emission.T * gains).T

    def _extract_params(self, params):
        # params are [tau, gains]
        return params[0], params[1:]

    def rel_gains(self, params) -> np.ndarray:
        _, gains = self._extract_params(params)
        return div_mean(gains)

    def _default_init_params(self, ndet: int) -> np.ndarray:
        return np.r_[0.01, np.ones(ndet)]

    def chi2_fun_jac(self, params, x, y) -> tuple[float, np.ndarray]:
        ypred = self.evaluate(params, x)
        resid = y - ypred
        chi2 = np.sum(resid**2)

        # gradient
        grad = np.empty_like(params)
        tau, gains = self._extract_params(params)

        # partial derivative of basis function wrt tau
        basis_partial_tau = (np.exp(-tau * x).T * gains).T * x
        grad[0] = np.sum(resid * basis_partial_tau)

        # partial derivative wrt g's
        grad[1:] = np.sum(resid * ypred, axis=-1) / gains

        return chi2, -2 * grad

    def hess(self, params, x, y) -> np.ndarray:
        tau, gains = self._extract_params(params)
        hess = np.zeros((params.size, params.size))
        h_diag = np.einsum("ii->i", hess)  # view of the diagonal

        # tau x tau
        basis_partial_tau = (np.exp(-tau * x).T * gains).T * x
        h_diag[0] = (basis_partial_tau**2).sum()

        # g x g
        # only the diagonal is non-zero
        basis_partial_g = -np.expm1(-tau * x)
        h_diag[1:] = np.einsum("ij,ij->i", basis_partial_g, basis_partial_g)

        # tau x g
        hess[0, 1:] = np.sum(basis_partial_tau * basis_partial_g, axis=-1)
        hess[1:, 0] = hess[0, 1:]

        return 2 * hess

    def hess_sp(self, params, x, y) -> sps.csr_matrix:
        tau, gains = self._extract_params(params)

        # tau x tau
        basis_partial_tau = (np.exp(-tau * x).T * gains).T * x
        tau_x_tau = (basis_partial_tau**2).sum()

        # g x g
        # only the diagonal is non-zero
        basis_partial_g = -np.expm1(-tau * x)
        g_x_g = np.einsum("ij,ij->i", basis_partial_g, basis_partial_g)

        # tau x g
        tau_x_g = np.sum(basis_partial_tau * basis_partial_g, axis=-1)

        # CSR format
        m = params.size
        nnz = 3 * m - 2

        # 1st row is full, others have 2 nonzeros each
        indptr = np.r_[0, np.arange(m, nnz + 1, 2)]

        indices = np.empty(nnz)
        indices[:m] = np.arange(m)  # 1st line
        indices[m::2] = 0  # 1st column
        indices[m + 1 :: 2] = np.arange(1, m)  # diagonal

        values = np.empty(nnz)
        values[0] = tau_x_tau
        values[1:m] = values[m::2] = tau_x_g
        values[m + 1 :: 2] = g_x_g

        return sps.csr_matrix((2 * values, indices, indptr), shape=(m, m))


class ExpModel2(Model):
    minimizer = "trust-constr"
    description = "exp $\\chi^2(\\tau,g_0,\\epsilon_i)$"
    plot_color = "tab:red"

    @staticmethod
    def param_labels(ndet: int) -> list[str]:
        return ["$\\tau$", "$g_0$"] + [
            "$\\epsilon_{" + str(x) + "}$" for x in range(ndet)
        ]

    def evaluate(self, params, x) -> np.ndarray:
        tau, mean_g, rel_g = self._extract_params(params)
        gains = mean_g * rel_g
        emission = -np.expm1(-tau * x)
        return (emission.T * gains).T

    def _extract_params(self, params):
        # params are [tau, mean_gain, eps_1, eps_2, ...]
        tau = params[0]
        mean_g = params[1]
        rel_g = 1 + params[2:]
        return tau, mean_g, rel_g

    def rel_gains(self, params) -> np.ndarray:
        _, _, rel_g = self._extract_params(params)
        return rel_g

    def _default_init_params(self, ndet: int) -> np.ndarray:
        return np.r_[0.01, 1, np.zeros(ndet)]

    def _minimize_kwargs(
        self, x, y, init_params, disp, tol, maxiter_factor, use_hess, sparse_hessian
    ):
        # get the "standard" kwargs from the parent class
        kwargs = super()._minimize_kwargs(
            x, y, init_params, disp, tol, maxiter_factor, use_hess, sparse_hessian
        )

        # add a constraint on the epsilons
        from scipy.optimize import LinearConstraint

        ndet = x.shape[0]
        kwargs["constraints"] = LinearConstraint(
            np.r_[0, 0, np.ones(ndet)].reshape(1, -1), lb=0, ub=0
        )
        return kwargs

    def chi2_fun_jac(self, params, x, y) -> tuple[float, np.ndarray]:
        ypred = self.evaluate(params, x)
        resid = y - ypred
        chi2 = np.sum(resid**2)

        # gradient
        grad = np.empty_like(params)
        tau, mean_g, rel_g = self._extract_params(params)
        gains = mean_g * rel_g

        # partial derivative of basis function wrt tau
        basis_partial_tau = (np.exp(-tau * x).T * gains).T * x
        grad[0] = np.sum(resid * basis_partial_tau)

        # basis_partial_g0 = y_pred / mean_g
        grad[1] = np.sum(resid * ypred) / mean_g

        # basis_partial_eps = y_pred / (1 + eps)
        grad[2:] = np.sum(resid * ypred, axis=-1) / rel_g

        return chi2, -2 * grad

    def hess(self, params, x, y) -> np.ndarray:
        tau, mean_g, rel_g = self._extract_params(params)
        gains = mean_g * rel_g
        hess = np.zeros((params.size, params.size))
        h_diag = np.einsum("ii->i", hess)  # view of the diagonal

        # intermediate quantities
        tau_el = tau * x
        ypred = (-np.expm1(-tau_el).T * gains).T

        # tau x tau
        basis_partial_tau = (np.exp(-tau_el).T * gains).T * x
        h_diag[0] = (basis_partial_tau**2).sum()

        # tau x mean_g and tau x ε's
        partial_sum_cross = np.sum(ypred * basis_partial_tau, axis=-1)
        hess[0, 1] = hess[1, 0] = partial_sum_cross.sum() / mean_g
        hess[0, 2:] = hess[2:, 0] = partial_sum_cross / rel_g
        del partial_sum_cross
        del basis_partial_tau

        # mean_g x mean_g
        partial_sum_square = np.sum(ypred**2, axis=-1)
        h_diag[1] = partial_sum_square.sum() / mean_g**2
        del ypred

        # mean_g x ε's
        hess[1, 2:] = hess[2:, 1] = partial_sum_square / gains

        # ε's x ε's
        # only the diagonal is non-zero
        h_diag[2:] = partial_sum_square / np.square(rel_g)
        del partial_sum_square

        return 2 * hess

    def hess_sp(self, params, x, y) -> sps.csr_matrix:
        tau, mean_g, rel_g = self._extract_params(params)
        gains = mean_g * rel_g

        # intermediate quantities
        tau_el = tau * x
        ypred = (-np.expm1(-tau_el).T * gains).T

        # tau x tau
        basis_partial_tau = (np.exp(-tau_el).T * gains).T * x
        tau_x_tau = (basis_partial_tau**2).sum()

        # tau x mean_g and tau x ε's
        partial_sum_cross = np.sum(ypred * basis_partial_tau, axis=-1)
        tau_x_mean_g = partial_sum_cross.sum() / mean_g
        tau_x_eps = partial_sum_cross / rel_g
        del partial_sum_cross
        del basis_partial_tau

        # mean_g x mean_g
        partial_sum_square = np.sum(ypred**2, axis=-1)
        mean_g_x_mean_g = partial_sum_square.sum() / mean_g**2
        del ypred

        # mean_g x ε's
        mean_g_x_eps = partial_sum_square / gains

        # ε's x ε's
        # only the diagonal is non-zero
        eps_x_eps = partial_sum_square / np.square(rel_g)
        del partial_sum_square

        # CSR format
        m = params.size
        nnz = 5 * m - 6

        # first 2 rows are full, others have 3 nonzeros each
        indptr = np.r_[0, m, np.arange(2 * m, nnz + 1, 3)]

        indices = np.empty(nnz)
        indices[:m] = np.arange(m)  # 1st line
        indices[m : 2 * m] = np.arange(m)  # 2nd line
        indices[2 * m :: 3] = 0  # 1st column
        indices[2 * m + 1 :: 3] = 1  # 2nd column
        indices[2 * m + 2 :: 3] = np.arange(2, m)  # diagonal

        values = np.empty(nnz)
        values[0] = tau_x_tau
        values[1] = values[m] = tau_x_mean_g
        values[2:m] = values[2 * m :: 3] = tau_x_eps
        values[m + 1] = mean_g_x_mean_g
        values[m + 2 : 2 * m] = values[2 * m + 1 :: 3] = mean_g_x_eps
        values[2 * m + 2 :: 3] = eps_x_eps

        return sps.csr_matrix((2 * values, indices, indptr), shape=(m, m))
