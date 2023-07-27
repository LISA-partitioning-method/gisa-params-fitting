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


def fit(Z):
    db = ProAtomDB.from_file("data/atoms.h5")
    db_record_dict = {
        1: (4, [0]),
        3: (6, [-2, -1, 0, 1, 2]),
        6: (6, [-2, -1, 0, 1, 2, 3]),
        7: (6, [2, -1, 0, 1, 2, 3]),
        8: (6, [-2, -1, 0, 1, 2, 3]),
        17: (9, [-2, -1, 0, 1, 2, 3]),
    }
    initial_guess_dict = {
        1: np.array([5.672, 1.505, 0.5308, 0.2204]),
        6: np.array([148.3, 42.19, 15.33, 6.146, 0.7846, 0.2511]),
        7: np.array([178.0, 52.42, 19.87, 1.276, 0.6291, 0.2857]),
        8: np.array([220.1, 65.66, 25.98, 1.685, 0.6860, 0.2311]),
        17: np.array(
            [0.0955, 0.2188, 0.5903, 0.7801, 8.8711, 19.2626, 164.0007, 373.7075, 591.4187]
        ),
    }

    opt_params = {
        1: (1e-5, 1000, 1e-5, 120, 1e-11),
        3: (1e-5, 1000, 1e-5, 120, 1e-11),
        6: (1e-5, 1000, 1e-5, 120, 1e-11),
        7: (1e-5, 1000, 1e-5, 120, 1e-11),
        8: (1e-5, 1000, 1e-5, 120, 1e-11),
        17: (1e-5, 1000, 1e-5, 120, 1e-10),
    }
    nprim = db_record_dict[Z][0]
    charges = db_record_dict[Z][1]

    records = []
    for Z, charge in zip([Z] * len(charges), charges):
        records.append(db.get_record(Z, charge))
    print("Number of records: ", len(records))

    def f(args):
        diff = 0.0
        Dk_array = np.zeros(len(records))
        for i, record in enumerate(records):
            weights = record.rgrid.weights
            radii = record.rgrid.radii
            nelec = record.number - record.charge

            def _g(g_args):
                rho_test, _ = get_proatom_rho(radii, g_args, args)
                return (rho_test - record.rho) * weights

            res = least_squares(
                _g,
                x0=[nelec / nprim] * nprim,
                bounds=opt_params[Z][2:4],
                ftol=opt_params[Z][-1],
                xtol=opt_params[Z][-1],
                gtol=opt_params[Z][-1],
            )
            Dk = res.x
            Dk_array[i] = np.sum(Dk)

            rho_test, _ = get_proatom_rho(radii, Dk, args)
            diff += (rho_test - record.rho) ** 2 * weights**2
            # diff += np.abs(rho_test - record.rho) * weights

        print_info("Sum of Dk for each record", Dk_array)
        return np.sqrt(diff)
        # return diff

    try:
        # res = least_squares(f, x0=[1] * nprim, bounds=(1e-2, 300))
        res = least_squares(
            f,
            # x0=np.linspace(1e-2, 200, nprim),
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

    x_fitted = res.x
    alphas_fitted = x_fitted

    print_info("alphas_fitted", np.sort(alphas_fitted))
    nb_elecs = []
    for i, record in enumerate(records):
        nb_elecs.append(record.number - record.charge)
    print_info("nb_elecs", nb_elecs)


if __name__ == "__main__":
    fit(17)
    # H: [ 0.2204  0.5308  1.5053  5.6722]
    # Li: [  0.008    0.042    0.1214   2.8398  10.7499  87.8413]
    # C: [  0.0656   0.2868   0.8043  15.6253  42.8325  99.5818]
    # N: [  0.1057   0.4141    1.1627   20.0579   51.6646  146.8978]
    # O: [  0.1564   0.5749    1.6097   22.8595   56.2564  184.0433]
    # Cl: [0.0955, 0.2188, 0.5903, 0.7801, 8.8711, 19.2626, 164.0007, 373.7075, 591.4187]
