#!/usr/bin/env python
import numpy as np
from scipy.optimize import least_squares

from general import ProModel

np.set_printoptions(precision=4, suppress=True, linewidth=np.inf)


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
        1: (1e-5, 1000, 1e-5, 120, 1e-10),
        3: (1e-5, 1000, 1e-5, 120, 1e-10),
        6: (1e-5, 1000, 1e-5, 120, 1e-10),
        7: (1e-5, 1000, 1e-5, 120, 1e-10),
        8: (1e-5, 1000, 1e-5, 120, 1e-10),
        17: (1e-5, 1000, 1e-5, 120, 1e-10),
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

    def obj_func_coeffs(par_coeffs):
        diff = 0.0
        charges = np.zeros(len(records))
        for i, record in enumerate(records):
            weights = record["weights"]
            points = record["points"]
            nelec = int(record["nelec"])

            def obj_func_pop(par_pops):
                pro_model = ProModel.from_pars(par_pops, par_coeffs)
                rho = pro_model.compute_proatom(points, None)
                # 4 * np.pi * r ** 2 has been included in weights
                return (rho - record["density"]) * weights

            def obj_func_pop_jac(par_pops):
                pro_model = ProModel.from_pars(par_pops, par_coeffs)
                rho_grad = pro_model.compute_derivatives(points, None)[::2, :] * weights
                return rho_grad.T

            x0 = [nelec / nprim] * nprim
            opt_pops = least_squares(
                obj_func_pop,
                x0=x0,
                jac=obj_func_pop_jac,
                bounds=opt_params[Z][2:4],
                ftol=opt_params[Z][-1],
                xtol=opt_params[Z][-1],
                gtol=opt_params[Z][-1],
            )

            ## use local `minimize` method
            # def obj_func_pop_scalar(par_pops):
            #     pro_model = ProModel.from_pars(par_pops, par_coeffs)
            #     rho = pro_model.compute_proatom(points, None)
            #     rho_diff = rho - record["density"]
            #     res = 0.5 * np.einsum("i,i", rho_diff**2, weights**2)
            #     return res
            #     # # Compute gradient
            #     # grad_Dk = pro_model.compute_derivatives(points, None)[::2, :]
            #     # grad = np.einsum("ni,i,i", grad_Dk, rho_diff, weights)
            #     # grad[grad == np.inf] = 1e10
            #     # np.nan_to_num(grad, copy=False)
            #     # return res, grad

            charges[i] = np.sum(opt_pops.x)
            pro_model = ProModel.from_pars(opt_pops.x, par_coeffs)
            rho_test = pro_model.compute_proatom(points, None)
            diff += (rho_test - record["density"]) ** 2 * weights**2

        print_info("Sum of Dk for atom: ", charges)
        return np.sqrt(diff)

    # TODO: how to add jac at this stage
    # def jac_matrix2(par_coeffs):
    #     pro_model = ProModel.from_pars(par_pops, par_coeffs)
    #     rho_grad = pro_model.compute_derivatives(radii, None)[::2, :] * weights
    #     return rho_grad.T

    opt_pars = least_squares(
        obj_func_coeffs,
        x0=initial_guess_dict.get(Z, np.linspace(1e-2, 1e2, nprim)),
        bounds=opt_params[Z][:2],
        ftol=opt_params[Z][-1],
        xtol=opt_params[Z][-1],
        gtol=opt_params[Z][-1],
        verbose=2,
    )

    ## use global `minimize` method
    # def cost_function(pars):
    #     par_coeffs = pars[:nprim]
    #     par_pops = pars[nprim:]
    #     cost = 0.0
    #     for i, record in enumerate(records):
    #         weights = record["weights"]
    #         points = record["points"]
    #         pro_model = ProModel.from_pars(par_pops, par_coeffs)
    #         rho = pro_model.compute_proatom(points, None)
    #         cost += np.sum(0.5 * ((rho - record["density"]) * weights) ** 2)
    #     return cost

    # pop_initial_vals = []
    # for elec in nb_elecs:
    #     # pop_initial_vals.extend([elec / nprim] * nprim)
    #     pop_initial_vals.extend([1.0] * nprim)
    # pop_initial_vals = np.asarray(pop_initial_vals)

    # coeffs_initial_vals = initial_guess_dict.get(Z, np.linspace(1e-2, 1e2, nprim))
    # x0 = np.concatenate((coeffs_initial_vals, pop_initial_vals))
    # print("pop initial values:")
    # print(pop_initial_vals)
    # print("coeffs initial values:")
    # print(coeffs_initial_vals)
    # A = np.zeros((len(nb_elecs), nprim + nprim * len(records)))
    # for idx, nelec in enumerate(nb_elecs):
    #     A[idx, nprim * (idx + 1) : nprim * (idx + 2)] = 1.0
    # print("matrix A")
    # print(A)
    # print("lb nad hb")
    # print(nb_elecs)
    # bounds = [tuple(opt_params[Z][:2])] * nprim + [tuple(opt_params[Z][2:4])] * len(
    #     nb_elecs
    # ) * nprim
    # print(bounds)

    # opt_pars = minimize(
    #     cost_function,
    #     x0,
    #     jac=False,
    #     method="trust-constr",
    #     bounds=bounds,
    #     tol=opt_params[Z][-1],
    #     constraints=LinearConstraint(A, nb_elecs, nb_elecs),
    #     # callback=callback,
    # )

    # for idx in range(len(nb_elecs)):
    #     opt_pop = opt_pars.x[nprim * (idx + 1) : nprim * (idx + 2)]
    #     print_info("optimized pop", opt_pop)
    #     print("The sum of optimized pop: ", np.sum(opt_pop))

    print_info("alphas_fitted", np.sort(opt_pars.x[:nprim])[::-1])
    print_info("nb_elecs", nb_elecs)


if __name__ == "__main__":
    fit(1)
    # H: [5.6774 1.5061 0.531  0.2204]
    # C: [133.3789  38.2891  14.1038   0.8005   0.3076   0.1036]
    # N: [173.6656  49.5977  18.7949   1.1963   0.4636   0.1496]
    # O: [229.1104  66.0381  24.8347   1.7326   0.7047   0.231 ]

    # Li: [60.3528 14.895   5.0545  1.9759  0.0971  0.0314]
    # Cl: [477.6906 154.6946  55.9616  19.7133  18.1434   8.9417   0.6404   0.3023   0.1364]
