import numpy as np


class BasisFunction1D:
    """Base class for atom-centered basis functions for the pro-molecular density.

    Each basis function instance stores also its parameters in ``self.pars``,
    which are always kept up-to-date. This simplifies the code a lot because
    the methods below can easily access the ``self.pars`` attribute when they
    need it, instead of having to rely on the caller to pass them in correctly.
    This is in fact a typical antipattern, but here it works well.
    """

    def __init__(self, pars, bounds):
        """Initialize a basis function.

        Parameters
        ----------
        pars
            The initial values of the proparameters for this function.
        bounds
            List of tuples with ``(lower, upper)`` bounds for each parameter.
            Use ``-np.inf`` and ``np.inf`` to disable bounds.

        """
        if len(pars) != len(bounds):
            raise ValueError("The number of parameters must equal the number of bounds.")
        self.pars = pars
        self.bounds = bounds

    @property
    def npar(self):
        """Number of parameters."""
        return len(self.pars)

    @property
    def population(self):
        """Population of this basis function."""
        raise NotImplementedError

    @property
    def population_derivatives(self):
        """Derivatives of the population w.r.t. proparameters."""
        raise NotImplementedError

    def get_cutoff_radius(self, density_cutoff):
        """Estimate the cutoff radius for the given density cutoff."""
        raise NotImplementedError

    def compute(self, points):
        """Compute the basisfunction values on a grid."""
        raise NotImplementedError

    def compute_derivatives(self, points):
        """Compute derivatives of the basisfunction values on a grid."""
        raise NotImplementedError


class ExponentialFunction1D(BasisFunction1D):
    """Exponential basis function for the MBIS pro density.

    See BasisFunction base class for API documentation.
    """

    def __init__(self, pars):
        if len(pars) != 2 and not (pars >= 0).all():
            raise TypeError("Expecting two positive parameters.")
        super().__init__(pars, [(5e-5, 1e2), (0.1, 1e3)])

    @property
    def population(self):
        return self.pars[0]

    @property
    def exponent(self):
        """Exponent of the exponential functions."""
        return self.pars[1]

    @property
    def population_derivatives(self):
        return np.array([1.0, 0.0])

    def get_cutoff_radius(self, density_cutoff):
        if density_cutoff <= 0.0:
            return np.inf
        population, exponent = self.pars
        return (np.log(population) - np.log(density_cutoff)) / exponent

    def _compute_dists(self, points, cache=None):
        return points

    def _compute_exp(self, exponent, dists, cache=None):
        return np.exp(-exponent * dists)

    def compute(self, points, cache=None):
        population, exponent = self.pars
        if exponent < 0 or population < 0:
            return np.full(len(points), np.inf)
        dists = self._compute_dists(points, cache)
        exp = self._compute_exp(exponent, dists, cache)
        prefactor = population * (exponent**3 / 8 / np.pi)
        return prefactor * exp

    def compute_derivatives(self, points, cache=None):
        population, exponent = self.pars
        if exponent < 0 or population < 0:
            return np.full((2, len(points)), np.inf)
        dists = self._compute_dists(points, cache)
        exp = self._compute_exp(exponent, dists, cache)
        factor = exponent**3 / 8 / np.pi
        vector = (population * exponent**2 / 8 / np.pi) * (3 - dists * exponent)
        return np.array([factor * exp, vector * exp])


class GaussianFunction1D(ExponentialFunction1D):
    """Gaussian basis function for the GISA pro density.

    See BasisFunction base class for API documentation.
    """

    def get_cutoff_radius(self, density_cutoff):
        if density_cutoff <= 0.0:
            return np.inf
        population, exponent = self.pars
        prefactor = population * (exponent / np.pi) ** 1.5
        if prefactor < 0 or prefactor < density_cutoff:
            return np.inf
        else:
            return np.sqrt((np.log(prefactor) - np.log(density_cutoff)) / exponent)

    def _compute_exp(self, exponent, dists, cache=None):
        return np.exp(-exponent * dists**2)

    def compute(self, points, cache=None):
        population, exponent = self.pars
        if exponent < 0 or population < 0:
            # return np.full(len(points), np.inf)
            return np.full(len(points), 1e100)
        dists = self._compute_dists(points, cache)
        exp = self._compute_exp(exponent, dists, cache)
        prefactor = population * (exponent / np.pi) ** 1.5
        return prefactor * exp

    def compute_derivatives(self, points, cache=None):
        population, exponent = self.pars
        if exponent < 0 or population < 0:
            return np.full((2, len(points)), np.inf)
        dists = self._compute_dists(points, cache)
        exp = self._compute_exp(exponent, dists, cache)
        factor = (exponent / np.pi) ** 1.5
        vector = (population * exponent**0.5 / np.pi**0.5) * (
            1.5 / np.pi - dists**2 * exponent / np.pi
        )
        return np.array([factor * exp, vector * exp])


class ProModelMeta(type):
    """Meta class for ProModel classes.

    This meta class registers all subclasses, making it easy to recreate a ProModel
    instance from the data stored in an NPZ file. Note that Python pickle files are
    not used for storing result because these are not suitable for long-term data
    preservation.

    """

    registry = {}

    def __new__(mcs, name, bases, namespace, **kwargs):
        result = super().__new__(mcs, name, bases, namespace, **kwargs)
        ProModelMeta.registry[name] = result
        return result


class ProModel(metaclass=ProModelMeta):
    """Base class for the promolecular density."""

    registry = {}

    def __init__(self, fns):
        """Initialize the prodensity model.

        Parameters
        ----------
        atnums
            Atomic number
        atchargs
            Atomic charge
        fns
            A list of basis functions, instances of ``BasisFunction``.
        """
        self.fns = fns

    @property
    def npar(self):
        return np.sum(fn.npar for fn in self.fns)

    def to_dict(self):
        """Return a dictionary representation of the pro-model, with with additional.

        Notes
        -----
        The primary purpose is to include sufficient information in the returned result
        to reconstruct this instance from the dictionary.

        It is recommended that subclasses try to include additional information that may
        be convenient for end users.

        All values in the dictionary must be np.ndarray instances.

        """
        return {
            "class": np.array(self.__class__.__name__),
            # "atnfns": atnfns,
            # "atnpars": atnpars,
            "propars": np.concatenate([fn.pars for fn in self.fns]),
        }

    @classmethod
    def from_dict(cls, data):
        """Create an instance of a ProModel subclass from a dictionary made with to_dict."""
        subcls = ProModelMeta.registry[str(data["class"])]
        if cls == subcls:
            raise TypeError("Cannot instantiate ProModel base class.")
        return subcls.from_dict(data)

    @property
    def population(self):
        """Promolecular population."""
        return sum(fn.population for fn in self.fns)

    def assign_pars(self, pars):
        """Assign the promolecule parameters to the basis functions."""
        ipar = 0
        for fn in self.fns:
            fn.pars[:] = pars[ipar : ipar + fn.npar]
            ipar += fn.npar

    def compute_proatom(self, points, cache=None):
        """Compute proatom density on a set of points.

        Parameters
        ----------
        points
            A set of points on which the proatom must be computed.
        cache
            An optional ComputeCache instance for reusing intermediate results.

        Returns
        -------
        pro
            The pro-atom density on the points of ``grid``.

        """
        pro = 0
        for fn in self.fns:
            pro += fn.compute(points, cache)
        return pro

    def get_cutoff_radii(self, density_cutoff):
        """Estimate the cutoff radii for all atoms."""
        radii = 0.0
        for fn in self.fns:
            radii = max(radii, fn.get_cutoff_radius(density_cutoff))
        return radii

    def pprint(self):
        """Print a table with the pro-parameters."""
        print(" ifn  atn       parameters...")
        for ifn, fn in enumerate(self.fns):
            print(
                "{:4d}  {:s}".format(
                    ifn,
                    " ".join(format(par, "15.8f") for par in fn.pars),
                )
            )

    def compute_derivatives2(self, points, weights, cache=None):
        gradient = np.zeros(self.npar, dtype=float)
        ipar = 0
        for _ifn, fn in enumerate(self.fns):
            fn_derivatives = fn.compute_derivatives(points, cache)
            gradient[ipar : ipar + fn.npar] = np.einsum("i,ji", weights, fn_derivatives)
            ipar += fn.npar
        return gradient

    def compute_derivatives(self, points, cache=None):
        gradient = np.zeros((self.npar, points.size), dtype=float)
        ipar = 0
        for _ifn, fn in enumerate(self.fns):
            fn_derivatives = fn.compute_derivatives(points, cache)
            # gradient[ipar : ipar + fn.npar] = np.einsum("i,ji", weights, fn_derivatives)
            gradient[ipar : ipar + fn.npar, :] = fn_derivatives
            ipar += fn.npar
        return gradient

    @classmethod
    def from_pars(cls, pops, coeffs):
        fns = [GaussianFunction1D((d, alpha)) for d, alpha in zip(pops, coeffs, strict=True)]
        return cls(fns)


class RadialGrid:
    def __init__(self, points, weights):
        self.points = points
        self.weights = weights

    def integrate(self, funcs):
        return np.einsum("i,i", self.weights, funcs)
