import os
import tempfile
import unittest

import numpy as np

packages_path = os.environ.get("COBAYA_PACKAGES_PATH") or os.path.join(
    tempfile.gettempdir(), "SPT_packages"
)


class SPT3GSystTest(unittest.TestCase):
    def setUp(self):
        from cobaya.install import install

        install({"likelihood": {"likelihoods.spt3g_syst": None}}, path=packages_path)

    def test_cobaya(self):
        """Test the Cobaya interface to the SPT3G lite likelihood."""
        from cobaya.model import get_model

        cosmo_params = dict(
            cosmomc_theta=0.010411,
            As=1e-10 * np.exp(3.1),
            ombh2=0.0221,
            omch2=0.1200,
            ns=0.96,
            Alens=1.0,
            tau=0.09,
        )

        fg_params = dict(
            kappa=0.0,
            Dl_Poisson_90x90=0.1,
            Dl_Poisson_90x150=0.1,
            Dl_Poisson_90x220=0.1,
            Dl_Poisson_150x150=0.1,
            Dl_Poisson_150x220=0.1,
            Dl_Poisson_220x220=0.1,
            TDust=19.6,
            ADust_TE_150=0.1647,
            BetaDust_TE=1.59,
            AlphaDust_TE=-2.42,
            ADust_EE_150=0.0236,
            BetaDust_EE=1.59,
            AlphaDust_EE=-2.42,
            mapTcal90=1.0,
            mapTcal150=1.0,
            mapTcal220=1.0,
            mapPcal90=1.0,
            mapPcal150=1.0,
            mapPcal220=1.0,
        )

        info = {
            "debug": True,
            "likelihood": {"likelihoods.spt3g_syst": None},
            "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}}},
            "params": {**cosmo_params, **fg_params},
            "modules": packages_path,
        }

        info["params"].update({f"yp{i}": 1.0 for i in range(10)})
        info["params"].update({f"bl{i}": 0.0 for i in range(10)})
        info["params"].update({f"ap{i}": 1.0 for i in range(10)})
        info["params"].update({f"dt{i}": 1.0 for i in range(10)})

        model = get_model(info)
        chi2 = -2 * model.loglike({})[0]
        print("chi2", chi2)
        self.assertAlmostEqual(chi2, 1160.1925138672325, 2)


if __name__ == "__main__":
    unittest.main()
