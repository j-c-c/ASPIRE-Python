"""
Miscellaneous Utilities that have no better place (yet).
"""
import hashlib
import importlib.resources
import logging
import sys
from itertools import chain, combinations

import numpy as np
from scipy.special import erf

from aspire.utils import grid_1d, grid_2d, grid_3d
from aspire.utils.rotation import Rotation

logger = logging.getLogger(__name__)


def importlib_path(package, resource):
    """
    Return the path to the resource as an actual file system path.
    Workaround importlib.resources deprecation of `path` in Python 3.11.
    This is expected to be safely removed after the minimal supported
    Python is 3.9.

    See ASPIRE-Python #546.

    :param package: Is either a name or a module object
        which conforms to the Package requirements.
    :param resource: Is the name of the resource to open within package;
        It may not contain path separators and it may not have sub-resources.
        (i.e. it cannot be a directory)
    :return: This function returns a context manager for use in a with statement.
        The context manager provides a pathlib.Path object.
    """

    py_version = sys.version_info

    # Use the deprecated method
    if py_version.major == 3 and py_version.minor < 9:
        p = importlib.resources.path(package, resource)
    else:
        p = importlib.resources.as_file(
            importlib.resources.files(package).joinpath(resource)
        )

    return p


def abs2(x):
    """
    Compute complex modulus squared.
    """

    return x.real**2 + x.imag**2


def powerset(iterable):
    """
    Generate all subsets of an iterable. Example:

    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)

    :return: Generator covering all subsets of iterable.
    """

    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def sha256sum(filename):
    """
    Return sha256 hash of filename.

    :param filename: path to file
    :return: sha256 hash as hex
    """

    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filename, "rb", buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])

    return h.hexdigest()


def gaussian_1d(size, mu=0, sigma=1, dtype=np.float64):
    """
    Returns the 1D Gaussian

    .. math::
        g(x)=\\exp\\left(\\frac{ -(x - \\mu)^2}{2\\sigma^2}\\right)

    in a 1D numpy array.

    :param size: The length of the returned array (pixels)
    :param mu: mean or center (pixels)
    :param sigma: standard deviation of the Gaussian
    :param dtype: dtype of returned array
    :return: Numpy array (1D)
    """

    # Construct centered mesh
    g = grid_1d(size, normalized=False, dtype=dtype)

    p = (g["x"] - mu) ** 2 / (2 * sigma**2)

    return np.exp(-p).astype(dtype, copy=False)


def gaussian_2d(size, mu=(0, 0), sigma=(1, 1), indexing="yx", dtype=np.float64):
    """
    Returns the 2D Gaussian

    .. math::
        g(x,y)=\\exp\\left(\\frac{-(x - \\mu_x)^2}{2\\sigma_x^2} +
                \\frac{-(y - \\mu_y)^2}{2\\sigma_y^2}\\right)

    in a square 2D numpy array.

    :param size: The length of each dimension of the returned array (pixels)
    :param mu: Iterable of len(2), :math:`(\\mu_x, \\mu_y)`, indicating the center of the Gaussian
    :param sigma: Iterable of len(2) or constant, :math:`(\\sigma_x, \\sigma_y)`, of the standard
            deviation in the x and y directions. A single value, :math:`\\sigma`, can be
            used when :math:`\\sigma_x = \\sigma_y`.
    :param indexing: The order of axis indexing, passed to `aspire.utils.grid_2d`
    :param dtype: dtype of returned array
    :return: Numpy array (2D)
    """
    if np.ndim(sigma) == 0:
        sigma = (sigma, sigma)

    mu = np.array(mu, dtype=dtype)
    sigma = np.array(sigma, dtype=dtype)
    if len(mu) != 2:
        raise ValueError("`mu` must be len(2).")
    if len(sigma) != 2:
        raise ValueError("`sigma` must be a scalar or len(2).")

    # Construct centered mesh
    g = grid_2d(size, shifted=False, normalized=False, indexing=indexing, dtype=dtype)

    if indexing == "yx":
        mu, sigma = mu[::-1], sigma[::-1]
        g["x"], g["y"] = g["y"], g["x"]
    elif indexing != "xy":
        raise ValueError("Indexing must be `yx` or `xy`.")

    p = (g["x"] - mu[0]) ** 2 / (2 * sigma[0] ** 2) + (g["y"] - mu[1]) ** 2 / (
        2 * sigma[1] ** 2
    )

    return np.exp(-p).astype(dtype, copy=False)


def gaussian_3d(size, mu=(0, 0, 0), sigma=(1, 1, 1), indexing="zyx", dtype=np.float64):
    """
    Returns the 3D Gaussian

    .. math::
        g(x,y,z)=\\exp\\left(\\frac{-(x - \\mu_x)^2}{2\\sigma_x^2} +
                \\frac{-(y - \\mu_y)^2}{2\\sigma_y^2} +
                \\frac{-(z - \\mu_z)^2}{2\\sigma_z^2}\\right)

    in a 3D numpy array.

    :param size: The length of each dimension of the returned array (pixels)
    :param mu: Iterable of len(3), :math:`(\\mu_x, \\mu_y, \\mu_z)`, indicating the center of the Gaussian
    :param sigma: Iterable of len(3) or constant, :math:`(\\sigma_x, \\sigma_y, \\sigma_z)`, of the standard deviation
            in the x, y, and z directions. A single value, :math:`\\sigma`, can be
            used when :math:`\\sigma_x = \\sigma_y = \\sigma_z`
    :param indexing: The order of axis indexing, passed to `aspire.utils.grid_3d`
    :param dtype: dtype of returned array
    :return: Numpy array (3D)
    """
    if np.ndim(sigma) == 0:
        sigma = (sigma, sigma, sigma)

    mu = np.array(mu, dtype=dtype)
    sigma = np.array(sigma, dtype=dtype)
    if len(mu) != 3:
        raise ValueError("`mu` must be len(3).")
    if len(sigma) != 3:
        raise ValueError("`sigma` must be a scalar or len(3).")

    # Construct centered mesh
    g = grid_3d(size, shifted=False, normalized=False, indexing=indexing, dtype=dtype)

    if indexing == "zyx":
        mu, sigma = mu[::-1], sigma[::-1]
        g["x"], g["y"], g["z"] = g["z"], g["y"], g["x"]
    elif indexing != "xyz":
        raise ValueError("Indexing must be `zyx` or `xyz`.")

    p = (
        (g["x"] - mu[0]) ** 2 / (2 * sigma[0] ** 2)
        + (g["y"] - mu[1]) ** 2 / (2 * sigma[1] ** 2)
        + (g["z"] - mu[2]) ** 2 / (2 * sigma[2] ** 2)
    )

    return np.exp(-p).astype(dtype, copy=False)


def bump_3d(size, spread=1, dtype=np.float64):
    """
    Returns a centered 3D bump function in a (size)x(size)x(size) numpy array.
    :param size: The length of the dimensions of the array (pixels.
    :param spread: A factor controling the spread of the bump function.
    :param dtype: dtype of returned array
    :return: Numpy array (3D)
    """
    g = grid_3d(size, dtype=dtype)
    selection = g["r"] < 1

    p = g["x"] ** 2 + g["y"] ** 2 + g["z"] ** 2

    bump = np.zeros((size,) * 3, dtype=dtype)
    bump[selection] = np.exp(-1 / (spread - spread * p[selection]))
    bump /= np.exp(-1 / spread)

    return bump


def circ(size, x0=0, y0=0, radius=1, peak=1, dtype=np.float64):
    """
    Returns a 2d `circ` function in a square 2d numpy array.

    where for r = sqrt(x**2 + y**2)

    circ(x,y) = peak : 0 <= r <= radius
                0 : otherwise

    Default is a centered circle of spread=peak=1.

    :param size: The height and width of returned array (pixels)
    :param x0: x cordinate of center (pixels)
    :param y0: y cordinate of center (pixels)
    :param radius: radius of circle
    :param peak: peak height at center
    :param dtype: dtype of returned array
    :return: Numpy array (2D)
    """

    # Construct centered mesh
    g = grid_2d(size, shifted=True, normalized=False, indexing="yx", dtype=dtype)

    vals = ((g["x"] - x0) ** 2 + (g["y"] - y0) ** 2) < radius * radius
    return (peak * vals).astype(dtype)


def inverse_r(size, x0=0, y0=0, peak=1, dtype=np.float64):
    """
    Returns a 2d inverse radius function in a square 2d numpy array.

    Where inverse_r(x,y): 1/sqrt(1 + x**2 + y**2)

    Default is a centered circle of peak=1.

    :param size: The height and width of returned array (pixels)
    :param x0: x cordinate of center (pixels)
    :param y0: y cordinate of center (pixels)
    :param peak: peak height at center
    :param dtype: dtype of returned array
    :return: Numpy array (2D)
    """

    # Construct centered mesh
    g = grid_2d(size, shifted=True, normalized=False, indexing="yx", dtype=dtype)

    # Compute the denominator
    vals = np.sqrt(1 + (g["x"] - x0) ** 2 + (g["y"] - y0) ** 2)

    return (peak / vals).astype(dtype)


def fuzzy_mask(L, r0, risetime, origin=None):
    """
    Create a centered 1D to 3D fuzzy mask of radius r0

    Made with an error function with effective rise time.

    :param L: The sizes of image in tuple structure
    :param r0: The specified radius
    :param risetime: The rise time for `erf` function
    :param origin: The coordinates of origin
    :return: The desired fuzzy mask
    """

    center = [sz // 2 + 1 for sz in L]
    if origin is None:
        origin = center

    grids = [np.arange(1 - org, ell - org + 1) for ell, org in zip(L, origin)]
    XYZ = np.meshgrid(*grids, indexing="ij")
    XYZ_sq = [X**2 for X in XYZ]
    R = np.sqrt(np.sum(XYZ_sq, axis=0))
    k = 1.782 / risetime
    m = 0.5 * (1 - erf(k * (R - r0)))

    return m


def all_pairs(n):
    """
    All pairs indexing (i,j) for i<j.

    :param n: The number of items to be indexed.
    :return: All n-choose-2 pairs (i,j), i<j.
    """
    pairs = [(i, j) for i in range(n) for j in range(n) if i < j]

    return pairs


def pairs_to_linear(n, i, j):
    """
    Converts from all_pairs indexing (i, j), where i<j, to linear indexing.
    ie. (0, 1) --> 0 and (n-2, n-1) --> n * (n - 1)/2 - 1
    """
    assert i < j < n, "i must be less than j, and both must be less than n."

    linear_index = n * (n - 1) // 2 - (n - i) * (n - i - 1) // 2 + j - i - 1

    return linear_index


def all_triplets(n):
    """
    All 3-tuples (i,j,k) where i<j<k.

    :param n: The number of items to be indexed.
    :returns: All 3-tuples (i,j,k), i<j<k.
    """
    triplets = [
        (i, j, k) for i in range(n) for j in range(n) for k in range(n) if i < j < k
    ]

    return triplets


def J_conjugate(A):
    """
    Conjugate the 3x3 matrix A by the diagonal matrix J=diag((-1, -1, 1)).

    :param A: A 3x3 matrix.
    :return: J*A*J
    """
    J = np.diag((-1, -1, 1))

    return J @ A @ J


def cyclic_rotations(order, dtype=np.float64):
    """
    Build all rotation matrices that rotate by multiples of 2pi/order about the z-axis.

    :param order: The order of cyclic symmetry
    :return: A Rotation object containing an (order)x3x3 array of rotation matrices.
    """
    angles = np.zeros((order, 3), dtype=dtype)
    angles[:, 2] = 2 * np.pi * np.arange(order) / order
    rots_symm = Rotation.from_euler(angles)

    return rots_symm
