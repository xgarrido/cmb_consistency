import argparse

from cobaya.run import run
from cobaya.yaml import yaml_load_file


def main():
    parser = argparse.ArgumentParser(description="A MCMC sampler for ACTPol likelihood")
    parser.add_argument(
        "-y",
        "--yaml-file",
        help="Yaml file holding sim/minization setup",
        default=None,
        required=True,
    )
    parser.add_argument(
        "--output-base-dir",
        help="Set the output base dir where to store results",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--leakage",
        help="Study T-E leakage",
        default=False,
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "--polareff",
        help="Study polar efficiency",
        default=False,
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "--ee-crap",
        help="Study EE crap",
        default=False,
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "--nparams",
        help="Number of parameter for polar eff. and TE leakage",
        default=10,
        required=False,
    )
    args = parser.parse_args()

    info = yaml_load_file(args.yaml_file)
    nbin = int(args.nparams)
    if args.leakage:
        info["params"].update({f"ap{i}": 1.0 for i in range(nbin)})
        info["params"].update({f"yp{i}": 1.0 for i in range(nbin)})
        info["params"].update(
            {f"bl{i}": {"prior": {"min": -0.1, "max": +0.1}, "proposal": 0.05} for i in range(nbin)}
        )
    elif args.polareff:
        info["params"].update(
            {f"yp{i}": {"prior": {"min": 0.5, "max": 1.5}, "proposal": 0.5} for i in range(nbin)}
        )
        info["params"].update({f"ap{i}": 1.0 for i in range(nbin)})
        info["params"].update({f"bl{i}": 0.0 for i in range(nbin)})
    elif args.ee_crap:
        info["params"].update(
            {f"ap{i}": {"prior": {"min": 0.5, "max": 1.5}, "proposal": 0.5} for i in range(nbin)}
        )
        info["params"].update({f"bl{i}": 0.0 for i in range(nbin)})
        info["params"].update({f"yp{i}": 1.0 for i in range(nbin)})
    else:
        info["params"].update({f"ap{i}": 1.0 for i in range(nbin)})
        info["params"].update({f"yp{i}": 1.0 for i in range(nbin)})
        info["params"].update({f"bl{i}": 0.0 for i in range(nbin)})

    if args.output_base_dir is not None:
        info["output"] = args.output_base_dir

    updated_info, sampler = run(info)


if __name__ == "__main__":
    main()
