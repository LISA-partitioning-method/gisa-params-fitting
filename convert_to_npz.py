#!/usr/bin/env python

from __future__ import print_function, division
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
    data = {
        'weights':weights, 
        'radii':radii, 
        'number':number, 
        'charge':charge,
        'nelec':nelec,
        'rho':rho
    }
    np.savez('atom_{}_{}.npz'.format(number, int(charge)), **data)


def main():
    db = ProAtomDB.from_file("data/atoms.h5")
    db_record_dict = {
        1: [0],
        3: [-2, -1, 0, 1, 2],
        6: [-2, -1, 0, 1, 2, 3],
        7: [-2, -1, 0, 1, 2, 3],
        8: [-2, -1, 0, 1, 2, 3],
        17: [-2, -1, 0, 1, 2, 3],
    }

    records = []
    for Z, charges in db_record_dict.items():
        for Z, charge in zip([Z] * len(charges), charges):
            record = db.get_record(Z, charge)
            to_npz(record)

main()
