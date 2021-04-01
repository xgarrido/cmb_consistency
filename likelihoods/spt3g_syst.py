from typing import Optional

import numpy as np
from spt3g_2020.spt3g import SPT3GPrototype


class spt3g_syst(SPT3GPrototype):
    def get_requirements(self):
        # State requisites to the theory code
        yp = {f"yp{i}": None for i in range(10)}
        bl = {f"bl{i}": None for i in range(10)}
        ap = {f"ap{i}": None for i in range(10)}
        dt = {f"dt{i}": None for i in range(10)}
        expected_fg_params = dict(
            kappa=None,
            Dl_Poisson_90x90=None,
            Dl_Poisson_90x150=None,
            Dl_Poisson_90x220=None,
            Dl_Poisson_150x150=None,
            Dl_Poisson_150x220=None,
            Dl_Poisson_220x220=None,
            TDust=None,
            ADust_TE_150=None,
            BetaDust_TE=None,
            AlphaDust_TE=None,
            ADust_EE_150=None,
            BetaDust_EE=None,
            AlphaDust_EE=None,
            mapTcal90=None,
            mapTcal150=None,
            mapTcal220=None,
            mapPcal90=None,
            mapPcal150=None,
            mapPcal220=None,
        )
        return {
            "Cl": {cl: self.lmax for cl in ["tt", "te", "ee"]},
            **expected_fg_params,
            **yp,
            **bl,
            **ap,
            **dt,
        }

    def loglike(self, dltt, dlte, dlee, **params_values):
        yp = np.repeat([v for k, v in params_values.items() if k.startswith("yp")], 6)[: self.nbins]
        bl = np.repeat([v for k, v in params_values.items() if k.startswith("bl")], 6)[: self.nbins]
        ap = np.repeat([v for k, v in params_values.items() if k.startswith("ap")], 6)[: self.nbins]
        dt = np.repeat([v for k, v in params_values.items() if k.startswith("dt")], 6)[: self.nbins]

        T_CMB = 2.72548  # CMB temperature
        h = 6.62606957e-34  # Planck's constant
        kB = 1.3806488e-23  # Boltzmann constant
        Ghz_Kelvin = h / kB * 1e9

        # Planck function normalised to 1 at nu0
        def Bnu(nu, nu0, T):
            return (
                (nu / nu0) ** 3
                * (np.exp(Ghz_Kelvin * nu0 / T) - 1)
                / (np.exp(Ghz_Kelvin * nu / T) - 1)
            )

        # Derivative of Planck function normalised to 1 at nu0
        def dBdT(nu, nu0, T):
            x0 = Ghz_Kelvin * nu0 / T
            x = Ghz_Kelvin * nu / T

            dBdT0 = x0 ** 4 * np.exp(x0) / (np.exp(x0) - 1) ** 2
            dBdT = x ** 4 * np.exp(x) / (np.exp(x) - 1) ** 2

            return dBdT / dBdT0

        lmin, lmax = self.lmin, self.lmax
        ells = np.arange(lmin, lmax + 2)
        dltt_cmb = dltt[lmin:lmax]
        dlte_cmb = dlte[lmin:lmax]
        dlee_cmb = dlee[lmin:lmax]

        dbs = np.empty_like(self.bandpowers)
        for i, (cross_spectrum, cross_frequency) in enumerate(
            zip(self.cross_spectra, self.cross_frequencies)
        ):
            dl_cmb = dlee if cross_spectrum == "EE" else dlte

            # Calculate derivatives for this position in parameter space.
            cl_derivative = dl_cmb[ells] * 2 * np.pi / (ells * (ells + 1))
            cl_derivative = 0.5 * (cl_derivative[2:] - cl_derivative[:-2])

            dls = np.zeros(len(self.ells))

            # Add super sample lensing
            # (In Cl space) SSL = -k/l^2 d/dln(l) (l^2Cl) = -k(l*dCl/dl + 2Cl)
            if self.super_sample_lensing:
                kappa = params_values.get("kappa")
                dls += -kappa * (
                    self.ells ** 2 * (self.ells + 1) / (2 * np.pi) * cl_derivative
                    + 2 * dl_cmb[self.ells]
                )

            # Aberration correction
            # AC = beta*l(l+1)dCl/dln(l)/(2pi)
            # Note that the CosmoMC internal aberration correction and the SPTpol Henning likelihood differ
            # CosmoMC uses dCl/dl, Henning et al dDl/dl
            # In fact, CosmoMC is correct:
            # https://journals-aps-org.eu1.proxy.openathens.net/prd/pdf/10.1103/PhysRevD.89.023003
            dls += (
                -self.aberration_coefficient
                * cl_derivative
                * self.ells ** 2
                * (self.ells + 1)
                / (2 * np.pi)
            )

            # Simple poisson foregrounds
            # This is any poisson power. Meant to describe both radio galaxies and DSFG. By giving each frequency combination an amplitude
            # to play with this gives complete freedom to the data
            if self.poisson_switch and cross_spectrum == "EE":
                Dl_poisson = params_values.get("Dl_Poisson_{}x{}".format(*cross_frequency))
                dls += self.ells * (self.ells + 1) * Dl_poisson / (3000 * 3001)

            # Polarised galactic dust
            if self.dust_switch:
                TDust = params_values.get("TDust")
                ADust = params_values.get(f"ADust_{cross_spectrum}_150")
                AlphaDust = params_values.get(f"AlphaDust_{cross_spectrum}")
                BetaDust = params_values.get(f"BetaDust_{cross_spectrum}")
                dfs = (
                    lambda beta, temp, nu0, nu: (nu / nu0) ** beta
                    * Bnu(nu, nu0, temp)
                    / dBdT(nu, nu0, T_CMB)
                )
                dust = ADust * (self.ells / 80) ** (AlphaDust + 2)
                for freq in cross_frequency:
                    dust *= dfs(BetaDust, TDust, 150, self.nu_eff_list.get(int(freq)))
                dls += dust

            # Scale by calibration
            if cross_spectrum == "EE":
                # Calibration for EE: 1/(Ecal_1*Ecal_2) since we matched the EE spectrum to Planck's
                calibration = 1.0
                for freq in cross_frequency:
                    calibration /= params_values.get(f"mapPcal{freq}")

                window = self.windows[:, i, :]
                db_cmb = ap * (
                    yp ** 2 * (window @ dlee_cmb)
                    + 2 * bl * (window @ dlte_cmb)
                    + bl ** 2 * (window @ dltt_cmb)
                )

            if cross_spectrum == "TE":
                # Calibration for TE: 0.5*(1/(Tcal_1*Ecal_2) + 1/(Tcal_2*Ecal_1))
                freq1, freq2 = cross_frequency
                calibration = 0.5 * (
                    1
                    / (params_values.get(f"mapTcal{freq1}") * params_values.get(f"mapPcal{freq2}"))
                    + 1
                    / (params_values.get(f"mapTcal{freq2}") * params_values.get(f"mapPcal{freq1}"))
                )
                window = self.windows[:, i, :]
                db_cmb = dt * (yp * (window @ dlte_cmb) + bl * (window @ dltt_cmb))

            # Binning via window and concatenate
            dbs[i * self.nbins : (i + 1) * self.nbins] = (self.windows[:, i, :] @ dls) * calibration
            dbs[i * self.nbins : (i + 1) * self.nbins] += db_cmb * calibration

        # Take the difference to the measured bandpower
        delta_cb = dbs - self.bandpowers

        # Construct the full covariance matrix
        cov_w_beam = self.cov + self.beam_cov * np.outer(dbs, dbs)

        chi2 = delta_cb @ np.linalg.inv(cov_w_beam) @ delta_cb
        sign, slogdet = np.linalg.slogdet(cov_w_beam)

        # Add calibration prior
        delta_cal = np.log(np.array([params_values.get(p) for p in self.calib_params]))
        cal_prior = delta_cal @ self.inv_calib_cov @ delta_cal

        self.log.debug(f"SPT3G X²/ndof = {chi2:.2f}/{len(delta_cb)}")
        self.log.debug(f"SPT3G detcov = {slogdet:.2f}")
        self.log.debug(f"SPT3G cal. prior = {cal_prior:.2f}")
        return -0.5 * (chi2 + slogdet + cal_prior)

    def logp(self, **data_params):
        Cls = self.provider.get_Cl(ell_factor=True)
        return self.loglike(Cls.get("tt"), Cls.get("te"), Cls.get("ee"), **data_params)
