#!/usr/bin/env python

from __future__ import print_function, division
import os
import numpy as np
from horton import *

np.set_printoptions(precision=4, suppress=True, linewidth=np.inf)


def to_npz(record):
    weights = record.rgrid.weights
    radii = record.rgrid.radii
    number = record.number
    charge = record.charge
    rho = record.rho
    nelec = number - charge
    res = np.einsum("i,i", weights, rho)
    # Note: 4*np.pi*r**2 has been included in weights.
    data = {
        "weights": weights,
        "points": radii,
        "atnums": np.array([number]),
        "charge": charge,
        "nelec": nelec,
        "density": rho,
        "atcoords": np.array([[0]], dtype=float),
    }
    np.savez("../denspart_atom_{}_{}.npz".format(number, int(charge)), **data)


def main():
    db = ProAtomDB.from_file("atoms.h5")
    db_record_dict = {
        1: [-2, -1, 0],
        3: [-2, -1, 0, 1, 2],
        5: [-2, -1, 0, 1, 2, 3],
        6: [-2, -1, 0, 1, 2, 3],
        7: [-2, -1, 0, 1, 2, 3],
        8: [-2, -1, 0, 1, 2, 3],
        9: [-2, -1, 0, 1, 2, 3],
        14: [-2, -1, 0, 1, 2, 3],
        16: [-2, -1, 0, 1, 2, 3],
        17: [-2, -1, 0, 1, 2, 3],
        35: [-2, -1, 0, 1, 2, 3],
    }

    records = []
    for Z, charges in db_record_dict.items():
        for Z, charge in zip([Z] * len(charges), charges):
            record = db.get_record(Z, charge)
            fname = "../denspart_atom_{}_{}.npz".format(Z, int(charge))
            if not os.path.exists(fname):
                to_npz(record)


main()
