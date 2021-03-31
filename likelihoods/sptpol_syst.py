from typing import Optional

import numpy as np
from sptpol_2017.sptpol import SPTPolPrototype


class sptpol_syst(SPTPolPrototype):
    def get_requirements(self):
        # State requisites to the theory code
        yp = {f"yp{i}": None for i in range(10)}
        bl = {f"bl{i}": None for i in range(10)}
        ap = {f"ap{i}": None for i in range(10)}
        dt = {f"dt{i}": None for i in range(10)}
        expected_fg_params = {
            "kappa": None,
            "czero_psTE_150": None,
            "czero_psEE_150": None,
            "ADust_TE": None,
            "ADust_EE": None,
            "alphaDust_TE": None,
            "alphaDust_EE": None,
            "mapTcal": None,
            "mapPcal": None,
            "beam1": None,
            "beam2": None,
        }
        return {
            "Cl": {cl: self.lmax for cl in ["tt", "te", "ee"]},
            **expected_fg_params,
            **yp,
            **bl,
            **ap,
            **dt,
        }

    def loglike(self, dltt, dlte, dlee, **params_values):
        # Getting foregrounds
        dlte_fg, dlee_fg = self.get_foregrounds(**params_values)

        # CMB from theory
        lmin, lmax = self.lmin, self.lmax
        dltt_cmb = dltt[lmin:lmax]
        dlte_cmb = dlte[lmin:lmax]
        dlee_cmb = dlee[lmin:lmax]

        # Calculate derivatives for this position in parameter space.
        dlte = dlte[lmin - 1 : lmax + 1]
        dlee = dlee[lmin - 1 : lmax + 1]
        rs_te = self.rawspec_factor * dlte
        rs_ee = self.rawspec_factor * dlee
        deriv_te = 0.5 * (rs_te[2:] - rs_te[:-2])
        deriv_ee = 0.5 * (rs_ee[2:] - rs_ee[:-2])

        # Subtract the kappa parameter for super sample lensing.
        # kappa parameter as described in Manzotti, et al. 2014, equation (32).
        kappa = params_values.get("kappa")
        dlte_fg -= kappa * deriv_te * self.cl_to_dl_conversion / self.ells
        dlee_fg -= kappa * deriv_ee * self.cl_to_dl_conversion / self.ells

        # Correct for aberration
        if self.correct_aberration:
            beta = 0.0012309
            dipole_cosine = -0.4033
            dlte_fg -= beta * dipole_cosine * self.ells * 0.5 * (dlte[2:] - dlte[:-2])
            dlee_fg -= beta * dipole_cosine * self.ells * 0.5 * (dlee[2:] - dlee[:-2])

        # Now bin into bandpowers with the window functions.
        yp = np.repeat([v for k, v in params_values.items() if k.startswith("yp")], 6)[: self.nbin]
        bl = np.repeat([v for k, v in params_values.items() if k.startswith("bl")], 6)[: self.nbin]
        ap = np.repeat([v for k, v in params_values.items() if k.startswith("ap")], 6)[: self.nbin]
        dt = np.repeat([v for k, v in params_values.items() if k.startswith("dt")], 6)[: self.nbin]
        win_te, win_ee = self.windows[: self.nbin], self.windows[self.nbin :]
        dbte = dt * (yp * (win_te @ dlte_cmb) + bl * (win_te @ dltt_cmb))
        dbee = ap * (
            yp ** 2 * (win_ee @ dlee_cmb)
            + 2 * bl * (win_ee @ dlte_cmb)
            + bl ** 2 * (win_ee @ dltt_cmb)
        )
        dbte += win_te @ dlte_fg
        dbee += win_ee @ dlee_fg

        # Scale theory spectrum by calibration:
        mapTcal = params_values.get("mapTcal")
        mapPcal = params_values.get("mapPcal")
        cal_TT = mapTcal ** 2
        cal_TE = mapTcal ** 2 * mapPcal
        cal_EE = mapTcal ** 2 * mapPcal ** 2
        dbte /= cal_TE
        dbee /= cal_EE

        # Beam errors
        beam1 = params_values.get("beam1")
        beam2 = params_values.get("beam2")
        beam_factor = (1 + self.beam_err[0] * beam1) * (1 + self.beam_err[1] * beam2)
        delta_cb = np.concatenate([dbte, dbee]) * beam_factor - self.spec

        chi2 = delta_cb @ self.invcov @ delta_cb

        self.log.debug(f"SPTPol X²/ndof = {chi2:.2f}/{self.nall}")
        return -0.5 * chi2  # + self.logp_const

    def logp(self, **data_params):
        Cls = self.provider.get_Cl(ell_factor=True)
        return self.loglike(Cls.get("tt"), Cls.get("te"), Cls.get("ee"), **data_params)
