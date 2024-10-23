# GISA parameter fitting

This Python project aims to fit parameters for the GISA (Gaussian Iterative Stockholder Analysis) method using spherically averaged atomic densities. The exponents, $\alpha_{a,k}$, of the Gaussian functions for each element in the GISA model are optimized using the least-squares method.

## Methodology

The optimization was performed for several atoms with varying charges, and the corresponding $\alpha_{a,k}$ values were determined for different atomic shells.

### Atomic data calculation

The atomic densities for different states of the elements (cation, neutral atom, and anion) are computed using GAUSSIAN16 at the `PBE0/6-311+G(d,p)` level of theory. The spherically averaged densities were computed using the [`Horton 2.1.1`](https://theochem.github.io/horton/2.1.1/index.html) Python package.

### Objective function

The fitted parameters optimize the exponents for a compact Gaussian s-type density basis set to accurately represent these spherically averaged densities. More details are presented below.

The spherical average of atomic density for element $a$ with charge $q$ at the origin (0,0,0) is defined as:
$$\langle \rho_{a,q} \rangle_s (r) = \int_{\Omega} \rho_a(\vec{r}) d\Omega$$
where $\Omega$ represents the angular-dependent coordinates.

The pro-atomic density is modeled as a sum of 1D Gaussian functions:
$$\rho_{a,q}^0 (r = \|\vec{r} - \vec{R}_a \|) = \sum_{k} c_{a,k} \exp^{-\alpha_{a,k} r^2}$$
where $\vec{R}_a = (0,0,0)$ is the atomic coordinate.

The objective function for element $a$ is defined as:
$$\sum_{q=-1, 0, 1} \| \langle \rho_{a,q} \rangle_s (r) -  \sum_{k} c_{a,k} \exp^{-\alpha_{a,k} r^2 } \|^2$$

This approach involves fitting isolated atomic data for 0, +1, and -1 charges using the least-squares method.

### Detailed algorithm for optimizing $\alpha_{a,k}$

The optimization of the Gaussian exponents $\alpha_{a,k}$ in this project involves the following key steps:

1. **Initial setup**:
   - For each element, atomic density data is obtained for its different charge states (cation, neutral, and anion). These densities are precomputed using a quantum chemistry package at a specific level of theory and stored in files. The data includes the densities, weights, and points that are used as the reference for optimization.
   - A predefined number of Gaussian basis functions, characterized by their exponents $\alpha_{a,k}$, is assigned for each element. An initial guess for these exponents is generated based on either previously optimized values or using a simple linear spacing over a specified range.

2. **Representation of pro-atomic density**:
   - The pro-atomic density is represented as a sum of Gaussian functions. Each function has an exponent $\alpha_{a,k}$ and a corresponding population coefficient $c_{a,k}$. These coefficients are initially guessed by dividing the total electron number of the atom evenly across the Gaussian functions.

3. **Objective function**:
   The goal is to minimize this difference across all charge states of the atom (neutral, cation, anion).

4. **Optimization process**:
   - The optimization process adjusts both the exponents $\alpha_{a,k}$ and the population coefficients $c_{a,k}$ using a least-squares fitting approach. This fitting method iteratively minimizes the objective function by finding the best combination of $\alpha_{a,k}$ and $c_{a,k}$ that produces a pro-atomic density closest to the actual atomic density.
   - For each iteration, the pro-atomic density is recalculated based on the current values of $\alpha_{a,k}$ and $c_{a,k}$. The difference between this computed density and the reference density is evaluated, and the parameters are updated to reduce this difference.

5. **Convergence criteria**:
   - The fitting process continues until the difference between the computed and reference densities falls below a predefined threshold, indicating that the optimization has converged. At this point, the final optimized values of the exponents $\alpha_{a,k}$ and population coefficients $c_{a,k}$ are obtained.

6. **Result output**:
   - After the optimization is complete, the optimized Gaussian exponents and coefficients are stored and used to model the atomic densities. These optimized parameters provide a compact and accurate representation of the atomic densities across different charge states.

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Running

The spherically averaged atomic densities are already collected in NumPy `NPZ` files in the `src` directory. Therefore, there is no need to run `Horton 2.1.1` to generate them.

```bash
cd src
python main.py
```

The `gauss.json` file is generated when the program finishes successfully.

It should be noted that the results may not be the same as those used in ``Horton-Part`` due to differences in the objective function.

### The exponents and initial values used in [Horton-Part](https://github.com/LISA-partitioning-method/horton-part)

The exponents for the Gaussian s-type density basis in both GISA and LISA methods (in atomic units).

| k  | H       | B        | C        | N        | O        | F        | Si       | S         | Cl        | Br        |
|----|---------|----------|----------|----------|----------|----------|----------|-----------|-----------|-----------|
| 1  | 5.6720  | 98.2299  | 148.3000 | 178.0000 | 220.1000 | 232.2846 | 366.5112 | 528.7272  | 622.3137  | 1027.3862 |
| 2  | 1.5050  | 27.7169  | 42.1900  | 52.4200  | 65.6600  | 73.1726  | 104.3665 | 147.5558  | 180.7931  | 84.3671   |
| 3  | 0.5308  | 9.7959   | 15.3300  | 19.8700  | 25.9800  | 30.0344  | 15.5123  | 17.6378   | 98.9482   | 67.8966   |
| 4  | 0.2204  | 0.5004   | 6.1460   | 1.2760   | 1.6850   | 2.4199   | 9.5104   | 17.5077   | 69.1275   | 64.9399   |
| 5  |         | 0.1942   | 0.7846   | 0.6291   | 0.6860   | 1.0096   | 7.8724   | 15.1251   | 20.2219   | 30.7992   |
| 6  |         | 0.0618   | 0.2511   | 0.2857   | 0.2311   | 0.3263   | 5.3849   | 7.1494    | 8.9831    | 6.4459    |
| 7  |         |          |          |          |          |          | 3.7020   | 0.5499    | 0.6418    | 5.3029    |
| 8  |         |          |          |          |          |          | 0.3241   | 0.2713    | 0.3052    | 4.4950    |
| 9  |         |          |          |          |          |          | 0.1076   | 0.1013    | 0.1370    | 2.6361    |
| 10 |         |          |          |          |          |          |          |           |           | 0.7183    |
| 11 |         |          |          |          |          |          |          |           |           | 0.3682    |
| 12 |         |          |          |          |          |          |          |           |           | 0.1390    |

The initial values for the Gaussian s-type density basis in both GISA and LISA methods (in atomic units).

| k  | H       | B        | C      | N      | O      | F       | Si       | S        | Cl        | Br        |
|----|---------|----------|--------|--------|--------|---------|----------|----------|-----------|-----------|
| 1  | 0.0429  | 0.1356   | 0.1330 | 0.1627 | 0.1869 | 0.2326  | 0.5063   | 0.4472   | 0.4127    | 1.4011    |
| 2  | 0.2639  | 0.6428   | 0.5955 | 0.6567 | 0.6576 | 0.7623  | 1.1758   | 1.1959   | 1.1066    | 0.0001    |
| 3  | 0.4790  | 1.0597   | 1.0749 | 0.9993 | 0.9751 | 0.8161  | 0.0001   | 0.0001   | 0.1210    | 0.0001    |
| 4  | 0.2127  | 1.9693   | 0.0202 | 2.3257 | 3.0657 | 2.9602  | 1.7484   | 0.0001   | 0.0001    | 6.6184    |
| 5  |         | 1.1509   | 2.7117 | 1.8949 | 2.5622 | 3.3411  | 0.4014   | 1.4710   | 0.9133    | 0.0001    |
| 6  |         | 0.0412   | 1.4779 | 0.9479 | 0.5528 | 0.8988  | 2.5315   | 6.0430   | 6.5025    | 0.0001    |
| 7  |         |          |        |        |        |         | 3.0395   | 4.2959   | 5.5666    | 16.7407   |
| 8  |         |          |        |        |        |         | 3.5767   | 2.2890   | 2.0125    | 0.0001    |
| 9  |         |          |        |        |        |         | 1.0358   | 0.2565   | 0.3716    | 1.0632    |
| 10 |         |          |        |        |        |         |          |          |           | 3.3418    |
| 11 |         |          |        |        |        |         |          |          |           | 5.0013    |
| 12 |         |          |        |        |        |         |          |          |           | 0.7444    |


## Issues

If you have any issues related to this project, please contact yxcheng2buaa@gmail.com.
