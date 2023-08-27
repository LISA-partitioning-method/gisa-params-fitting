#!/usr/bin/env python

import numpy as np
from scipy.optimize import least_squares

np.set_printoptions(precision=4, suppress=True, linewidth=np.inf)


def get_pro_a_k(D_k, alpha_k, r, nderiv=0):
    """The primitive Gaussian function with exponent of `alpha_k`."""
    f = D_k * (alpha_k / np.pi) ** 1.5 * np.exp(-alpha_k * r**2)
    if nderiv == 0:
        return f
    elif nderiv == 1:
        return -2 * alpha_k * r * f
    else:
        raise NotImplementedError


def get_proatom_rho(r, prefactors, alphas):
    """Get proatom density for atom `iatom`."""
    nprim = len(prefactors)
    # r = rgrid.radii
    y = np.zeros(len(r), float)
    d = np.zeros(len(r), float)
    for k in range(nprim):
        D_k, alpha_k = prefactors[k], alphas[k]
        y += get_pro_a_k(D_k, alpha_k, r, 0)
        d += get_pro_a_k(D_k, alpha_k, r, 1)
    return y, d


def print_info(name, value):
    """
    A function to print a formatted output.
    """
    print(f"{name}")
    print(value)
    print("*" * 80)


def get_record(Z, charge):
    return np.load(f"denspart_atom_{Z}_{int(charge)}.npz")


def fit(Z):
    db_record_dict = {
        1: (4, [0]),
        # 3: (6, [-2, -1, 0, 1, 2]),
        # 6: (6, [-2, -1, 0, 1, 2, 3]),
        # 7: (6, [-2, -1, 0, 1, 2, 3]),
        # 8: (6, [-2, -1, 0, 1, 2, 3]),
        # 17: (9, [-2, -1, 0, 1, 2, 3]),
        3: (6, [-1, 0, 1]),
        6: (6, [-1, 0, 1]),
        7: (6, [-1, 0, 1]),
        8: (6, [-1, 0, 1]),
        9: (6, [-1, 0, 1]),
        14: (9, [-1, 0, 1]),
        16: (9, [-1, 0, 1]),
        17: (9, [-1, 0, 1]),
        35: (12, [-1, 0, 1]),
    }
    initial_guess_dict = {
        # 1: np.array([5.672, 1.505, 0.5308, 0.2204]),
        # 6: np.array([148.3, 42.19, 15.33, 6.146, 0.7846, 0.2511]),
        # 7: np.array([178.0, 52.42, 19.87, 1.276, 0.6291, 0.2857]),
        # 8: np.array([220.1, 65.66, 25.98, 1.685, 0.6860, 0.2311]),
    }

    opt_params = {
        1: (1e-5, 1000, 1e-5, 120, 1e-8),
        3: (1e-5, 1000, 1e-5, 120, 1e-8),
        6: (1e-5, 1000, 1e-5, 120, 1e-8),
        7: (1e-5, 1000, 1e-5, 120, 1e-8),
        8: (1e-5, 1000, 1e-5, 120, 1e-8),
        9: (1e-5, 1000, 1e-5, 120, 1e-8),
        14: (1e-5, 1e8, 1e-5, 120, 1e-10),
        16: (1e-5, 1e8, 1e-5, 120, 1e-10),
        17: (1e-5, 1e8, 1e-5, 120, 1e-10),
        35: (1e-5, 1e8, 1e-5, 120, 1e-10),
    }
    nprim = db_record_dict[Z][0]
    charges = db_record_dict[Z][1]

    records = []
    for atnum, charge in zip([Z] * len(charges), charges, strict=True):
        record = get_record(atnum, charge)
        records.append(record)
    print("Number of records: ", len(records))

    def outer_loop(args):
        diff = 0.0
        Dk_array = np.zeros(len(records))
        for i, record in enumerate(records):
            weights = record["weights"]
            radii = record["points"]
            nelec = int(record["nelec"])

            def inner_loop(g_args):
                rho_test, _ = get_proatom_rho(radii, g_args, args)
                # 4 * np.pi * r ** 2 has been included in weights
                return np.abs(rho_test - record["density"]) * weights

            res = least_squares(
                inner_loop,
                x0=[nelec / nprim] * nprim,
                bounds=opt_params[Z][2:4],
                ftol=opt_params[Z][-1],
                xtol=opt_params[Z][-1],
                gtol=opt_params[Z][-1],
            )
            Dk = res.x
            Dk_array[i] = np.sum(Dk)

            rho_test, _ = get_proatom_rho(radii, Dk, args)
            diff += (rho_test - record["density"]) ** 2 * weights**2

        print_info(f"Sum of Dk for record with {nelec} electrons", Dk_array)
        return np.sqrt(diff)

    try:
        res = least_squares(
            outer_loop,
            x0=initial_guess_dict.get(Z, np.linspace(1e-2, 1e2, nprim)),
            bounds=opt_params[Z][:2],
            ftol=opt_params[Z][-1],
            xtol=opt_params[Z][-1],
            gtol=opt_params[Z][-1],
            verbose=2,
        )
    except RuntimeError as e:
        print("Error occurred in least_squares_fit: ", e)
        return

    nb_elecs = []
    for _i, record in enumerate(records):
        nb_elecs.append(int(record["nelec"]))

    print_info("alphas_fitted", np.sort(res.x)[::-1])
    print_info("nb_elecs", nb_elecs)


if __name__ == "__main__":
    fit(14)
    # H: [5.6774 1.5061 0.531  0.2204]
    # C: [133.3789  38.2891  14.1038   0.8005   0.3076   0.1036]
    # N: [173.6656  49.5977  18.7949   1.1963   0.4636   0.1496]
    # O: [229.1104  66.0381  24.8347   1.7326   0.7047   0.231 ]

    # Li: [60.3528 14.895   5.0545  1.9759  0.0971  0.0314]
    # F: [319.6233  85.5603  31.7878   2.4347   1.0167   0.3276]
    # Si: [374.1708 118.753   69.9363   9.0231   8.3656   4.0225   0.3888   0.2045   0.0711]
    # Si new: [592.0697 213.778   88.4554  71.7604   8.3302   4.0274   0.4191   0.2271   0.076 ]
    # S: [715.0735 240.4533 120.1752  14.3796   7.0821   0.5548   0.5176   0.2499   0.1035]
    # Cl: [1139.4058  379.1381  151.8297   34.9379   19.5054    8.9484    0.6579    0.3952  0.1635]
    # Br: [1026.4314, 118.9876, 115.0694, 82.964, 66.2115, 64.8824, 6.4775, 5.2615, 1.6828, 0.5514,
    # 0.2747, 0.1136,]
