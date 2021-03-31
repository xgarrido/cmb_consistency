import os
import tempfile
import unittest

import numpy as np

packages_path = os.environ.get("COBAYA_PACKAGES_PATH") or os.path.join(
    tempfile.gettempdir(), "SPT_packages"
)


class SPTPolSystTest(unittest.TestCase):
    def setUp(self):
        from cobaya.install import install

        install({"likelihood": {"likelihoods.sptpol_syst": None}}, path=packages_path)

    def test_cobaya(self):
        """Test the Cobaya interface to the SPTPol lite likelihood."""
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
            czero_psTE_150=0.0,
            czero_psEE_150=0.0837416,
            ADust_TE=0.1647,
            ADust_EE=0.0236,
            alphaDust_TE=-2.42,
            alphaDust_EE=-2.42,
            mapTcal=1.0,
            mapPcal=1.0,
            beam1=0.0,
            beam2=0.0,
        )

        info = {
            "debug": True,
            "likelihood": {"likelihoods.sptpol_syst": None},
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
        self.assertAlmostEqual(chi2, 162.98103875445057, 5)


if __name__ == "__main__":
    unittest.main()
