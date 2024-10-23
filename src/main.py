#!/usr/bin/env python
import numpy as np
from scipy.optimize import least_squares

from general import ProModel

np.set_printoptions(precision=4, suppress=True, linewidth=np.inf)
np.random.seed = 10


def print_info(name, value):
    """
    A function to print a formatted output.
    """
    print(f"{name}")
    print(value)
    print("*" * 80)


def get_record(Z, charge):
    return np.load(f"denspart_atom_{Z}_{int(charge)}.npz")


def opt_pop(record, points, weights, coeffs, pop0, pop_bounds, ftol, xtol, gtol):
    def f(par_pops):
        pro_model = ProModel.from_pars_gauss(par_pops, coeffs)
        rho = pro_model.compute_proatom(points, None)
        # 4 * np.pi * r ** 2 has been included in weights
        return (rho - record["density"]) * weights

    opt_pops = least_squares(f, x0=pop0, bounds=pop_bounds, ftol=ftol, xtol=xtol, gtol=gtol)
    return opt_pops.x


def opt_coeff(records, pop_bounds, coeff0, coeff_bounds, ftol, xtol, gtol):
    def f(par_coeffs):
        diff = 0.0
        charges = np.zeros(len(records))
        for i, record in enumerate(records):
            weights, points = record["weights"], record["points"]
            nelec = int(record["nelec"])

            nprim = len(par_coeffs)
            pop0 = [nelec / nprim] * nprim

            opt_pops = opt_pop(
                record, points, weights, par_coeffs, pop0, pop_bounds, ftol, xtol, gtol
            )

            model = ProModel.from_pars_gauss(opt_pops, par_coeffs)
            rho_test = model.compute_proatom(points)

            rho = record["density"]
            charges[i] = np.sum(opt_pops)
            diff += (rho_test - rho) ** 2 * weights**2

        print_info("Sum of Dk for atom: ", charges)
        return np.sqrt(diff)

    opt_pars = least_squares(
        f, x0=coeff0, bounds=coeff_bounds, ftol=ftol, xtol=xtol, gtol=gtol, verbose=2
    )

    return opt_pars.x


def fit(Z):
    db_record_dict = {
        1: (4, [0]),
        3: (6, [-1, 0, 1]),
        5: (6, [-1, 0, 1]),
        6: (6, [-1, 0, 1]),
        7: (6, [-1, 0, 1]),
        8: (6, [-1, 0, 1]),
        9: (6, [-1, 0, 1]),
        14: (9, [-1, 0, 1]),
        16: (9, [-1, 0, 1]),
        17: (9, [-1, 0, 1]),
        35: (12, [-1, 0, 1]),
    }

    initial_guess_dict = {}

    known_guess_dict = {
        1: np.array([5.672, 1.505, 0.5308, 0.2204]),
        6: np.array([148.3, 42.19, 15.33, 6.146, 0.7846, 0.2511]),
        7: np.array([178.0, 52.42, 19.87, 1.276, 0.6291, 0.2857]),
        8: np.array([220.1, 65.66, 25.98, 1.685, 0.6860, 0.2311]),
    }

    opt_params = {
        1: (1e-5, 1000, 1e-5, 120, 1e-8),
        3: (1e-5, 1000, 1e-5, 120, 1e-8),
        5: (1e-5, 1000, 1e-5, 120, 1e-8),
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
    nb_elecs = []
    for atnum, charge in zip([Z] * len(charges), charges, strict=True):
        record = get_record(atnum, charge)
        records.append(record)
        elec = int(record["nelec"])
        nb_elecs.append(elec)
    print("Number of records: ", len(records))

    coeff_bounds = opt_params[Z][:2]
    pop_bounds = opt_params[Z][2:4]
    ftol = xtol = gtol = opt_params[Z][-1]
    if Z in known_guess_dict:
        coeffs = known_guess_dict[Z]
    else:
        coeff0 = initial_guess_dict.get(Z, np.linspace(1e-2, 1e2, nprim))
        coeffs = opt_coeff(records, pop_bounds, coeff0, coeff_bounds, ftol, xtol, gtol)

    record = get_record(Z, 0)
    weights = record["weights"]
    points = record["points"]
    pop0 = [Z / nprim] * nprim

    initial_pop = opt_pop(record, points, weights, coeffs, pop0, pop_bounds, ftol, xtol, gtol)
    print_info("Optimized exp coeffs: ", coeffs)
    print_info("nb_elecs", nb_elecs)
    print("Optimized initial pops: ")
    print(initial_pop)

    return np.asarray([2.0] * nprim), coeffs, initial_pop


if __name__ == "__main__":
    import json

    data = {}
    for Z in [1, 3, 5, 6, 7, 8, 9, 14, 16, 17, 35]:
        orders, coeffs, pops = fit(Z)
        combined = sorted(zip(coeffs, pops, strict=False), key=lambda x: x[0], reverse=True)
        sorted_coeffs, sorted_pops = zip(*combined, strict=False)
        sorted_coeffs = [round(c, 4) for c in sorted_coeffs]
        sorted_pops = [round(c, 4) for c in sorted_pops]
        data[Z] = [list(orders), list(sorted_coeffs), list(sorted_pops)]

    with open("gauss.json", "w") as f:
        json.dump(data, f, indent=4)
