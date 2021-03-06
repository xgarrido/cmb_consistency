import numpy as np


def test_cobaya():
    """Test the Cobaya interface to the SPTPol lite likelihood."""
    from cobaya.model import get_model
    from cobaya.yaml import yaml_load

    info_yaml = f"""
        debug: true
        likelihood:
            likelihoods.sptpol_lite:

        theory:
            camb:
                extra_args:
                    lens_potential_accuracy: 1

        params:
            cosmomc_theta: 0.010411
            As: {1e-10 * np.exp(3.1)}
            ombh2: 0.0221
            omch2: 0.1200
            ns: 0.96
            Alens: 1.0
            tau: 0.09
    """
    info = yaml_load(info_yaml)
    info["params"].update({f"yp{i}": 1.0 for i in range(10)})
    info["params"].update({f"bl{i}": 0.0 for i in range(10)})
    info["params"].update({f"ap{i}": 1.0 for i in range(10)})
    fg_params = {
        "kappa": 0.0,
        "czero_psTE_150": 0.0,
        "czero_psEE_150": 0.0837416,
        "ADust_TE": 0.1647,
        "ADust_EE": 0.0236,
        "alphaDust_TE": -2.42,
        "alphaDust_EE": -2.42,
        "mapTcal": 1.0,
        "mapPcal": 1.0,
        "beam1": 0.0,
        "beam2": 0.0,
    }
    info["params"].update(**fg_params)
    model = get_model(info)

    chi2 = -2 * model.loglike({})[0]
    print("chi2", chi2)
    assert np.isclose(chi2, 161.5146486050572)


if __name__ == "__main__":
    test_cobaya()
