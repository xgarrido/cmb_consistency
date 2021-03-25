import numpy as np
from cobaya.likelihoods.base_classes import PlanckPlikLite


class planck_lite_syst(PlanckPlikLite):
    def get_requirements(self):
        # State requisites to the theory code
        yp = {f"yp{i}": None for i in range(20)}
        bl = {f"bl{i}": None for i in range(20)}
        ap = {f"ap{i}": None for i in range(20)}
        dt = {f"dt{i}": None for i in range(20)}

        return {**super().get_requirements(), "A_planck": None, **yp, **bl, **ap, **dt}

    def get_chi_squared(self, L0, ctt, cte, cee, calPlanck=1, yp=1.0, bl=0.0, ap=1.0, dt=1.0):
        cl = np.empty(self.used_indices.shape)
        ix = 0
        for tp, cell in enumerate([ctt, cte, cee]):
            for i in self.used_bins[tp]:
                cl[ix] = np.dot(
                    cell[self.blmin[i] - L0 : self.blmax[i] - L0 + 1],
                    self.weights[self.blmin[i] : self.blmax[i] + 1],
                )
                if tp == 1:  # TE
                    cltt = np.dot(
                        ctt[self.blmin[i] - L0 : self.blmax[i] - L0 + 1],
                        self.weights[self.blmin[i] : self.blmax[i] + 1],
                    )
                    cl[ix] = dt[i] * (yp[i] * cl[ix] + bl[i] * cltt)
                if tp == 2:  # EE
                    clte = np.dot(
                        cte[self.blmin[i] - L0 : self.blmax[i] - L0 + 1],
                        self.weights[self.blmin[i] : self.blmax[i] + 1],
                    )
                    cltt = np.dot(
                        ctt[self.blmin[i] - L0 : self.blmax[i] - L0 + 1],
                        self.weights[self.blmin[i] : self.blmax[i] + 1],
                    )
                    cl[ix] = ap[i] * (yp[i] ** 2 * cl[ix] + 2 * bl[i] * clte + bl[i] ** 2 * cltt)
                ix += 1
        cl /= calPlanck ** 2
        diff = self.X_data - cl
        return self._fast_chi_squared(self.invcov, diff)

    def logp(self, **data_params):
        Cls = self.provider.get_Cl(ell_factor=True)
        yp = np.repeat([v for k, v in data_params.items() if k.startswith("yp")], 10)
        bl = np.repeat([v for k, v in data_params.items() if k.startswith("bl")], 10)
        ap = np.repeat([v for k, v in data_params.items() if k.startswith("ap")], 10)
        dt = np.repeat([v for k, v in data_params.items() if k.startswith("dt")], 10)

        return -0.5 * self.get_chi_squared(
            0,
            Cls.get("tt"),
            Cls.get("te"),
            Cls.get("ee"),
            data_params[self.calibration_param],
            yp,
            bl,
            ap,
            dt,
        )
