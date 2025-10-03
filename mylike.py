"""
.. module:: likelihoods.gaussian_mixture

:Synopsis: Gaussian mixture likelihood
:Author: Jesus Torrado

"""
# Global
from typing import Any
import numpy as np
from scipy.stats import norm, uniform, random_correlation
from scipy.special import logsumexp

# Local
from cobaya.likelihood import Likelihood
from cobaya.log import LoggedError
from cobaya.mpi import share_mpi, is_main_process
from cobaya.typing import InputDict, Union, Sequence
from cobaya.functions import inverse_cholesky


class MyLike(Likelihood):
    """
    Gaussian likelihood.
    """
    file_base_name = 'mylike'

    # yaml variables
    mean: float
    stdev: float
    
    #input_params_prefix: str
    #output_params_prefix: str

    def initialize(self):
        self.log.info("Initializing")
        super().initialize()
        self._input_params_order = self.input_params


    def initialize_with_params(self):
        """
        Initializes the gaussian distributions.
        """
        self.log.info("Initializing")
        self._input_params_order = self._input_params_order or self.input_params
        self.gaussian = norm(loc=self.mean, scale=self.stdev)

    def logp(self, **params_values):
        """
        Computes the log-likelihood for a given set of parameters.
        """
        x = params_values['x']
        return self.gaussian.logpdf(x)
        


# Scripts to generate random means and covariances #######################################

def random_mean(ranges, n_modes=1, mpi_warn=True, random_state=None):
    """
    Returns a uniformly sampled point (as an array) within a list of bounds ``ranges``.

    The output of this function can be used directly as the value of the option ``mean``
    of the :class:`likelihoods.gaussian`.

    If ``n_modes>1``, returns an array of such points.
    """
    if not is_main_process() and mpi_warn:
        print("WARNING! "
              "Using with MPI: different process will produce different random results.")
    mean = np.array([uniform.rvs(loc=r[0], scale=r[1] - r[0], size=n_modes,
                                 random_state=random_state) for r in ranges])
    mean = mean.T
    if n_modes == 1:
        mean = mean[0]
    return mean


def random_cov(ranges, O_std_min=1e-2, O_std_max=1, n_modes=1,
               mpi_warn=True, random_state=None):
    """
    Returns a random covariance matrix, with standard deviations sampled log-uniformly
    from the length of the parameter ranges times ``O_std_min`` and ``O_std_max``, and
    uniformly sampled correlation coefficients between ``rho_min`` and ``rho_max``.

    The output of this function can be used directly as the value of the option ``cov`` of
    the :class:`likelihoods.gaussian`.

    If ``n_modes>1``, returns a list of such matrices.
    """
    if not is_main_process() and mpi_warn:
        print("WARNING! "
              "Using with MPI: different process will produce different random results.")
    dim = len(ranges)
    scales = np.array([r[1] - r[0] for r in ranges])
    cov = []
    for _ in range(n_modes):
        stds = scales * 10 ** (uniform.rvs(size=dim, loc=np.log10(O_std_min),
                                           scale=np.log10(O_std_max / O_std_min),
                                           random_state=random_state))
        this_cov = np.diag(stds).dot(
            (random_correlation.rvs(dim * stds / sum(stds), random_state=random_state)
             if dim > 1 else np.eye(1)).dot(np.diag(stds)))
        # Symmetrize (numerical noise is usually introduced in the last step)
        cov += [(this_cov + this_cov.T) / 2]
    if n_modes == 1:
        cov = cov[0]
    return cov


def info_random_gaussian_mixture(ranges, n_modes=1, input_params_prefix="",
                                 output_params_prefix="", O_std_min=1e-2, O_std_max=1,
                                 derived=False, mpi_aware=True,
                                 random_state=None, add_ref=False):
    """
    Wrapper around ``random_mean`` and ``random_cov`` to generate the likelihood and
    parameter info for a random Gaussian.

    If ``mpi_aware=True``, it draws the random stuff only once, and communicates it to
    the rest of the MPI processes.

    If ``add_ref=True`` (default: False) adds a reference pdf for the input parameters,
    provided that the gaussian mixture is unimodal (otherwise raises ``ValueError``).
    """
    cov: Any
    mean: Any
    if is_main_process() or not mpi_aware:
        cov = random_cov(ranges, n_modes=n_modes, O_std_min=O_std_min,
                         O_std_max=O_std_max, mpi_warn=False, random_state=random_state)
        if n_modes == 1:
            cov = [cov]
        # Make sure it stays away from the edges
        mean = [[]] * n_modes
        for i in range(n_modes):
            std = np.sqrt(cov[i].diagonal())
            factor = 3
            ranges_mean = [[r[0] + factor * s, r[1] - +factor * s] for r, s in
                           zip(ranges, std)]
            # If this implies min>max, take the centre
            ranges_mean = [
                (r if r[0] <= r[1] else 2 * [(r[0] + r[1]) / 2]) for r in ranges_mean]
            mean[i] = random_mean(ranges_mean, n_modes=1, mpi_warn=False,
                                  random_state=random_state)
    else:
        mean, cov = None, None
    if mpi_aware:
        mean, cov = share_mpi((mean, cov))
    dimension = len(ranges)
    info: InputDict = {"likelihood": {"gaussian_mixture": {
        "means": mean, "covs": cov, "input_params_prefix": input_params_prefix,
        "output_params_prefix": output_params_prefix, "derived": derived}},
        "params": dict(
            # sampled
            tuple((input_params_prefix + "_%d" % i,
                   {"prior": {"min": ranges[i][0], "max": ranges[i][1]},
                    "latex": r"\alpha_{%i}" % i})
                  for i in range(dimension)) +
            # derived
            (tuple((output_params_prefix + "_%d" % i,
                    {"latex": r"\beta_{%i}" % i})
                   for i in range(dimension * n_modes)) if derived else ()))}
    if add_ref:
        if n_modes > 1:
            raise ValueError("Cannot add a good reference pdf ('add_ref=True') for "
                             "multimodal distributions")
        for i, (p, v) in enumerate(info["params"].items()):
            v["ref"] = {"dist": "norm", "loc": mean[0][i], "scale": np.sqrt(cov[0][i, i])}
    return info
