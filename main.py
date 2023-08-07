#!/usr/bin/env python

from __future__ import print_function, division
import numpy as np
from scipy.optimize import least_squares
from horton import *

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
    print("{}".format(name))
    print(value)
    print("*" * 80)


def get_record(Z, charge):
    return np.load('atom_{}_{}.npz'.format(Z, int(charge)))



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
        17: (9, [-1, 0, 1]),
    }
    initial_guess_dict = {
        # 1: np.array([5.672, 1.505, 0.5308, 0.2204]),
        # 6: np.array([148.3, 42.19, 15.33, 6.146, 0.7846, 0.2511]),
        # 7: np.array([178.0, 52.42, 19.87, 1.276, 0.6291, 0.2857]),
        # 8: np.array([220.1, 65.66, 25.98, 1.685, 0.6860, 0.2311]),
        # 17: np.array(
        #     [0.0955, 0.2188, 0.5903, 0.7801, 8.8711, 19.2626, 164.0007, 373.7075, 591.4187]
        # ),
    }

    opt_params = {
        1: (1e-5, 1000, 1e-5, 120, 1e-8),
        3: (1e-5, 1000, 1e-5, 120, 1e-8),
        6: (1e-5, 1000, 1e-5, 120, 1e-8),
        7: (1e-5, 1000, 1e-5, 120, 1e-8),
        8: (1e-5, 1000, 1e-5, 120, 1e-8),
        17: (1e-5, 1000, 1e-5, 120, 1e-10),
    }
    nprim = db_record_dict[Z][0]
    charges = db_record_dict[Z][1]

    records = []
    for Z, charge in zip([Z] * len(charges), charges):
        record = get_record(Z, charge)
        records.append(record)
    print("Number of records: ", len(records))

    def outer_loop(args):
        diff = 0.0
        Dk_array = np.zeros(len(records))
        for i, record in enumerate(records):
            weights = record['weights']
            radii = record['radii']
            nelec = int(record['nelec'])

            def inner_loop(g_args):
                rho_test, _ = get_proatom_rho(radii, g_args, args)
                # return np.abs(rho_test - record['rho']) * radii**2
                # return np.abs(rho_test - record['rho']) * radii**2 * weights
                return np.abs(rho_test - record['rho']) * weights

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
            # diff += (rho_test - record['rho']) ** 2 * radii**4
            # diff += (rho_test - record['rho']) ** 2 * weights**2 * radii**4
            diff += (rho_test - record['rho']) ** 2 * weights**2

        print_info("Sum of Dk for record with {} electrons".format(nelec), Dk_array)
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
        print("Error occurred in curve_fit: ", e)
        return

    nb_elecs = []
    for i, record in enumerate(records):
        nb_elecs.append(int(record['nelec']))

    print_info("alphas_fitted", np.sort(res.x)[::-1])
    print_info("nb_elecs", nb_elecs)


if __name__ == "__main__":
    fit(1)
    # H: [5.6774 1.5061 0.531  0.2204]
    # C: [133.3789  38.2891  14.1038   0.8005   0.3076   0.1036]
    # N: [173.6656  49.5977  18.7949   1.1963   0.4636   0.1496]
    # O: [229.1104  66.0381  24.8347   1.7326   0.7047   0.231 ]

    # Li: [60.3528 14.895   5.0545  1.9759  0.0971  0.0314]
    # Cl: [477.6906 154.6946  55.9616  19.7133  18.1434   8.9417   0.6404   0.3023   0.1364]
