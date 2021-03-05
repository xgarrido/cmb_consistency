"""
.. module:: _sptpol_lite_prototype

:Synopsis: Definition of python-native CMB likelihood for SPTPol TT+TE
:Author: Xavier Garrido

"""

# Global
import os
from typing import Optional, Sequence

import numpy as np
from cobaya.conventions import _packages_path
from cobaya.likelihoods._base_classes import _InstallableLikelihood


class _sptpol_lite_prototype(_InstallableLikelihood):
    install_options = {
        "download_url": "https://lambda.gsfc.nasa.gov/data/suborbital/SPT/sptpol_2017/sptpol_cosmomc_nov16_v1p3.tar.gz",
        "data_path": "sptpol_2017_lite",
    }

    nbin: Optional[int] = 56
    nfreq: Optional[int] = 1
    windows_lmin: Optional[int] = 3
    windows_lmax: Optional[int] = 10600
    use_cl: Sequence[str] = ["tt", "te", "ee"]
    correct_aberration: Optional[bool] = True

    data_folder: Optional[str] = "sptpol_2017_lite/sptpol_500d_TEEE"
    bp_file: Optional[str]
    cov_file: Optional[str]
    window_dir: Optional[str]
    beam_file: Optional[str]

    def initialize(self):
        # Set path to data
        if (not getattr(self, "path", None)) and (not getattr(self, _packages_path, None)):
            raise LoggedError(
                self.log,
                f"No path given to SPTPol data. Set the likelihood property 'path' or "
                "the common property '{_packages_path}'.",
            )
        # If no path specified, use the modules path
        data_file_path = os.path.normpath(
            getattr(self, "path", None) or os.path.join(self.packages_path, "data")
        )

        self.data_folder = os.path.join(data_file_path, self.data_folder)
        if not os.path.exists(self.data_folder):
            raise LoggedError(
                self.log,
                f"The 'data_folder' directory does not exist. Check the given path [self.data_folder].",
            )

        if self.nfreq != 1:
            raise LoggedError(self.log, "Sorry, current code wont work for multiple freqs")
        if self.windows_lmin < 2 or self.windows_lmin >= self.windows_lmax:
            raise LoggedError(self.log, "Invalid ell ranges for SPTPol")

        bands_per_freq = 3  # Should be three for SPTpol (TT,TE,EE, although mostly ignore TT).
        self.nband = bands_per_freq * self.nfreq
        self.nall = self.nbin * self.nfreq * (bands_per_freq - 1)  # Cov doesn't know about TT.

        # Read in bandpowers
        # Should be TE, EE, TT, in that order from SPTpol analysis.
        dummy, self.spec = np.loadtxt(os.path.join(self.data_folder, self.bp_file), unpack=True)
        # self.spec = spec.reshape((self.nband, self.nbin))

        # Read in covariance
        # Should be TE, EE
        cov = np.fromfile(os.path.join(self.data_folder, self.cov_file))
        self.cov = cov.reshape((self.nall, self.nall))

        # Read in windows
        # Should be TE, EE
        self.windows = np.array(
            [
                np.loadtxt(
                    os.path.join(self.data_folder, self.window_dir, f"window_{i}"), unpack=True
                )[1]
                for i in range(1, self.nall + 1)
            ]
        )

        # Get beam error term
        n_beam_terms = 2
        dummy, beam_err = np.loadtxt(os.path.join(self.data_folder, self.beam_file), unpack=True)
        self.beam_err = beam_err.reshape((n_beam_terms, self.nall))

        self.lmax = self.windows_lmax
        self.ells = np.arange(self.windows_lmin - 1, self.windows_lmax + 1)
        self.cl_to_dl_conversion = (self.ells * (self.ells + 1)) / (2 * np.pi)
        for var in ["nbin", "nfreq", "nall", "windows_lmin", "windows_lmax", "data_folder", "lmax"]:
            self.log.debug(f"{var} = {getattr(self, var)}")

    def get_requirements(self):
        # State requisites to the theory code
        # yp = {f"yp{i}": None for i in range(10)}
        # bl = {f"bl{i}": None for i in range(10)}
        # ap = {f"ap{i}": None for i in range(10)}
        # return {**yp, **bl, **ap, "Cl": {cl: self.lmax for cl in self.use_cl}}
        return dict(Cl={cl: self.lmax for cl in self.use_cl})

    def get_chi_squared(self, cov, res):
        from scipy.linalg import cho_factor, cho_solve

        L, low = cho_factor(cov)

        # compute det
        detcov = 2.0 * np.sum(np.log(np.diag(L)))

        # Compute C-1.d
        invCd = cho_solve((L, low), res)

        # Compute chi2
        chi2 = res @ invCd

        return chi2, detcov

    def get_foregrounds(self, dlte, dlee, **params_values):

        # Calculate derivatives for this position in parameter space.
        # kappa parameter as described in Manzotti, et al. 2014, equation (32).
        rawspec_factor = self.ells ** 2 / self.cl_to_dl_conversion
        delta_cl_te = 0.5 * np.diff(rawspec_factor * dlte[self.ells]) / self.ells[1:]
        delta_cl_ee = 0.5 * np.diff(rawspec_factor * dlee[self.ells]) / self.ells[1:]

        # First get model foreground spectrum (in Cl).
        # Note all the foregrounds are recorded in Dl at l=3000, so we
        # divide by d3000 to get to a normalized Cl spectrum.
        #
        # Start with Poisson power and subtract the kappa parameter for super sample lensing.
        d3000 = 3000 * 3001 / (2 * np.pi)
        poisson_level_TE = params_values.get("czero_psTE_150") / d3000
        poisson_level_EE = params_values.get("czero_psEE_150") / d3000
        kappa = params_values.get("kappa")
        dlte_fg = (poisson_level_TE - kappa * delta_cl_te) * self.cl_to_dl_conversion[1:]
        dlee_fg = (poisson_level_EE - kappa * delta_cl_ee) * self.cl_to_dl_conversion[1:]

        # Add dust foreground model (defined in Dl)
        ADust_TE = params_values.get("ADust_TE")
        ADust_EE = params_values.get("ADust_EE")
        alphaDust_TE = params_values.get("alphaDust_TE")
        alphaDust_EE = params_values.get("alphaDust_EE")
        dlte_fg += ADust_TE * (self.ells[1:] / 80) ** (alphaDust_TE + 2)
        dlee_fg += ADust_EE * (self.ells[1:] / 80) ** (alphaDust_EE + 2)

        return dlte_fg, dlee_fg

    def loglike(self, dltt, dlte, dlee, **params_values):
        yp = np.repeat([v for k, v in params_values.items() if k.startswith("yp")], 10)
        bl = np.repeat([v for k, v in params_values.items() if k.startswith("bl")], 10)
        ap = np.repeat([v for k, v in params_values.items() if k.startswith("ap")], 10)

        # Getting foregrounds
        dlte_fg, dlee_fg = self.get_foregrounds(dlte, dlee, **params_values)

        # CMB from theory
        dlte_cmb = dlte[self.windows_lmin : self.windows_lmax + 1]
        dlee_cmb = dlee[self.windows_lmin : self.windows_lmax + 1]

        if self.correct_aberration:
            beta = 0.0012309
            dipole_cosine = -0.4033
            dlte_cmb += -beta * dipole_cosine * self.ells[1:] * np.diff(dlte[self.ells]) / 2
            dlee_cmb += -beta * dipole_cosine * self.ells[1:] * np.diff(dlee[self.ells]) / 2

        # Now bin into bandpowers with the window functions.
        dbte = self.windows[: self.nbin] @ (dlte_fg + dlte_cmb)
        dbee = self.windows[self.nbin :] @ (dlee_fg + dlee_cmb)

        # Scale theory spectrum by calibration:
        mapTcal = params_values.get("mapTcal")
        mapPcal = params_values.get("mapPcal")
        cal_TT = mapTcal ** 2
        cal_TE = mapTcal ** 2 * mapPcal
        cal_EE = mapTcal ** 2 * mapPcal ** 2
        dbte /= cal_TE
        dbee /= cal_EE

        # # Beam errors
        beam1 = params_values.get("beam1")
        beam2 = params_values.get("beam2")
        beam_factor = (1 + self.beam_err[0] * beam1) * (1 + self.beam_err[1] * beam2)
        delta_cb = np.concatenate([dbte, dbee]) * beam_factor - self.spec[self.nbin :]

        chi2, detcov = self.get_chi_squared(self.cov, delta_cb)
        self.log.debug(f"SPTPol X² = {chi2:.2f}")
        return -0.5 * (chi2 + detcov)

    def logp(self, **data_params):
        Cls = self.provider.get_Cl(ell_factor=True)
        return self.loglike(Cls.get("tt"), Cls.get("te"), Cls.get("ee"), **data_params)


class sptpol_lite(_sptpol_lite_prototype):
    r"""
    SPTpol 2017 500d EETE power spectrum 50 < ell < 8000 (Henning et al 2017)
    """
    pass
