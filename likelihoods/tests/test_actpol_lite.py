import likelihoods as lk
import numpy as np


def get_example_spectra():
    filename = lk.ACTPowerSpectrumData.data_dir + "bf_ACTPol_WMAP_lcdm.minimum.theory_cl"
    tt_lmax = 5000
    ell, dell_tt, dell_te, dell_ee = np.genfromtxt(
        filename,
        delimiter=None,
        unpack=True,
        max_rows=tt_lmax - 1,
        usecols=(0, 1, 2, 3),
    )
    return ell, dell_tt, dell_te, dell_ee


def test_bmin():
    # nonzero bmin
    ell, dell_tt, dell_te, dell_ee = get_example_spectra()
    like = lk.ACTPowerSpectrumData(bmin=24)
    chi2 = -2 * like.loglike(dell_tt, dell_te, dell_ee, 1.003)
    print("ACTPol chi2 = " + "{0:.12f}".format(chi2))
    print("Expected:     235.146031846935")
    assert np.isclose(chi2, 235.146031846935)


def test_single_channel():
    """This function tests out the single channels functionality of this likelihood code."""

    ell, dell_tt, dell_te, dell_ee = get_example_spectra()

    # TT only
    like = lk.ACTPowerSpectrumData(use_tt=True, use_te=False, use_ee=False)
    chi2 = -2 * like.loglike(dell_tt, dell_te, dell_ee, 1.003)
    print("ACTPol chi2(TT) = {0:.12f}".format(chi2))
    assert np.isclose(chi2, 97.4331220842641)

    # TE only
    like = lk.ACTPowerSpectrumData(use_tt=False, use_te=True, use_ee=False)
    chi2 = -2 * like.loglike(dell_tt, dell_te, dell_ee, 1.003)
    print("ACTPol chi2(TE) = {0:.12f}".format(chi2))
    assert np.isclose(chi2, 81.6194890026420)

    # EE only
    like = lk.ACTPowerSpectrumData(use_tt=False, use_te=False, use_ee=True)
    chi2 = -2 * like.loglike(dell_tt, dell_te, dell_ee, 1.003)
    print("ACTPol chi2(EE) = {0:.12f}".format(chi2))
    assert np.isclose(chi2, 98.5427508626497)

    # TE+EE only
    like = lk.ACTPowerSpectrumData(use_tt=False, use_te=True, use_ee=True)
    chi2 = -2 * like.loglike(dell_tt, dell_te, dell_ee, 1.003)
    print("ACTPol chi2(TE+EE) = {0:.12f}".format(chi2))
    assert np.isclose(chi2, 188.252270007375)

    # TT+TE+EE only
    like = lk.ACTPowerSpectrumData(use_tt=True, use_te=True, use_ee=True)
    chi2 = -2 * like.loglike(dell_tt, dell_te, dell_ee, 1.003)
    print("ACTPol chi2(TT+TE+EE) = {0:.12f}".format(chi2))
    assert np.isclose(chi2, 288.252869629064)


def test_deep_wide_field():
    ell, dell_tt, dell_te, dell_ee = get_example_spectra()

    # TT wide only
    like = lk.ACTPowerSpectrumData(use_tt=True, use_te=False, use_ee=False, use_deep=False)
    chi2 = -2 * like.loglike(dell_tt, dell_te, dell_ee, 1.003)
    print(f"ACTPol chi2(TT) = {chi2:.12f} (wide only)")
    assert np.isclose(chi2, 40.610845288918)

    # TT deep only
    like = lk.ACTPowerSpectrumData(use_tt=True, use_te=False, use_ee=False, use_wide=False)
    chi2 = -2 * like.loglike(dell_tt, dell_te, dell_ee, 1.003)
    print(f"ACTPol chi2(TT) = {chi2:.12f} (deep only)")
    assert np.isclose(chi2, 58.408151760399)

    # TT+TE+EE wide only
    like = lk.ACTPowerSpectrumData(use_deep=False)
    chi2 = -2 * like.loglike(dell_tt, dell_te, dell_ee, 1.003)
    print(f"ACTPol chi2(TT+TE+EE) = {chi2:.12f} (wide only)")
    assert np.isclose(chi2, 146.653901865858)

    # TT+TE+EE deep only
    like = lk.ACTPowerSpectrumData(use_wide=False)
    chi2 = -2 * like.loglike(dell_tt, dell_te, dell_ee, 1.003)
    print(f"ACTPol chi2(TT+TE+EE) = {chi2:.12f} (deep only)")
    assert np.isclose(chi2, 143.100396999979)


def test_cobaya():
    """Test the Cobaya interface to the ACT likelihood."""
    # cobaya_installed = pytest.importorskip("cobaya", minversion="3.0")
    from cobaya.model import get_model
    from cobaya.yaml import yaml_load

    info_yaml = r"""
        likelihood:
            likelihoods.ACTPol_lite_DR4:
                components:
                    - tt
                    - te
                    - ee
                lmax: 6000

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
            yp2: 1.0
        """
    info = yaml_load(info_yaml)
    model = get_model(info)
    chi2 = -2 * model.loglike({"ns": 1.0, "H0": 70})[0]
    print("chi2", chi2)
    assert np.isfinite(chi2)


if __name__ == "__main__":
    test_single_channel()
    test_bmin()
    test_deep_wide_field()
    test_cobaya()
