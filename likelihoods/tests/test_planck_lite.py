import numpy as np


def test_cobaya():
    """Test the Cobaya interface to the Planck lite likelihood."""
    from cobaya.model import get_model
    from cobaya.yaml import yaml_load

    info_yaml = r"""
        likelihood:
            likelihoods.planck_lite_syst:
              dataset_params:
                use_cl: tt te ee

        theory:
            camb:
                extra_args:
                    lens_potential_accuracy: 1

        params:
            ns: 1.0
            H0: 70
            A_planck: 1.0
        """
    info = yaml_load(info_yaml)
    info["params"].update({f"yp{i}": 1.0 for i in range(20)})
    info["params"].update({f"bl{i}": 0.0 for i in range(20)})
    info["params"].update({f"ap{i}": 1.0 for i in range(20)})
    info["params"].update({f"dt{i}": 1.0 for i in range(20)})
    model = get_model(info)
    chi2 = -2 * model.loglike({})[0]
    print("chi2", chi2)
    assert np.isclose(chi2, 4625.883691714649)


if __name__ == "__main__":
    test_cobaya()
