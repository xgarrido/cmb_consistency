import likelihoods as lk
import numpy as np


def test_cobaya():
    """Test the Cobaya interface to the ACT likelihood."""
    from cobaya.model import get_model
    from cobaya.yaml import yaml_load

    info_yaml = r"""
        likelihood:
            likelihoods.planck_lite:
              dataset_params:
                use_cl: tt te ee

        theory:
            camb:
                extra_args:
                    lens_potential_accuracy: 1

        params:
            ns:
                prior:
                  min: 0.8
                  max: 1.2
            H0:
                prior:
                  min: 40
                  max: 100
            A_planck: 1.0
        """
    info = yaml_load(info_yaml)
    info["params"].update({f"yp{i}": {"prior": {"min": 0.5, "max": 1.5}} for i in range(20)})
    info["params"].update({f"bl{i}": {"prior": {"min": 0.5, "max": 1.5}} for i in range(20)})
    info["params"].update({f"ap{i}": {"prior": {"min": 0.5, "max": 1.5}} for i in range(20)})
    model = get_model(info)
    yp = {f"yp{i}": 1.0 for i in range(20)}
    bl = {f"bl{i}": 0.0 for i in range(20)}
    ap = {f"ap{i}": 0.0 for i in range(20)}
    chi2 = -2 * model.loglike({"ns": 1.0, "H0": 70, "A_planck": 1.0, **yp, **bl, **ap})[0]
    print("chi2", chi2)
    assert np.isclose(chi2, 4625.883691714649)


if __name__ == "__main__":
    test_cobaya()
