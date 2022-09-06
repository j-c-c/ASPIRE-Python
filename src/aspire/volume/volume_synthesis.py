import abc

import numpy as np
from numpy.linalg import qr

from aspire.utils import Rotation, bump_3d, grid_3d
from aspire.utils.random import Random, randn
from aspire.volume import Volume


class SyntheticVolumeBase(abc.ABC):
    def __init__(self, L, C, symmetry_type, seed=None, dtype=np.float64):
        self.L = L
        self.C = C
        self.symmetry_type = symmetry_type
        self.seed = seed
        self.dtype = dtype

    @abc.abstractmethod
    def generate(self):
        """
        Called to generate and return the synthetic volumes.

        Each concrete subclass should implement this.
        """

    @abc.abstractmethod
    def _check_symmetry(self):
        """
        Called to check that volumes are instantiated with compatible symmetry type.

        Each concrete subclass should implement this.
        """

    def __repr__(self):
        return f"{self.__dict__}"


class LegacyVolume(SyntheticVolumeBase):
    """
    Legacy ASPIRE Volume constructed of 3D Gaussian blobs.

    Suffers from too large point variances.
    Included for migration of legacy unit tests.
    """

    def __init__(self, L, C=2, symmetry_type=None, K=16, seed=None, dtype=np.float64):
        """
        :param L: Resolution of the Volume(s) in pixels.
        :param C: Number of Volumes to generate.
        :param symmetry_type: Must be None for LegacyVolume.
        :param K: Number of Gaussian blobs used to construct the Volume(s).
        :param seed: Random seed for generating random Gaussian blobs.
        :param dtype: dtype for Volume(s).
        """
        super().__init__(L, C, symmetry_type, seed=seed, dtype=dtype)
        self.K = K
        self._check_symmetry()

    def generate(self):
        """
        Called to generate and return an ASPIRE LegacyVolume as a Volume instance.
        """
        return gaussian_blob_vols(
            L=self.L,
            C=self.C,
            symmetry_type=self.symmetry_type,
            seed=self.seed,
            dtype=self.dtype,
        )

    def _check_symmetry(self):
        """
        Checks that `symmetry_type` is set to None.
        """
        if self.symmetry_type is not None:
            raise ValueError("symmetry_type must be None for LagacyVolume")


class CompactVolume(SyntheticVolumeBase):
    """
    A LegacyVolume or CnSymmetricVolume that has compact support within the unit sphere.
    """

    def __init__(self, L, C, symmetry_type, K=16, seed=None, dtype=np.float64):
        """
        :param L: Resolution of the Volume(s) in pixels.
        :param C: Number of Volumes to generate.
        :param symmetry_type: Volume symmetry. None or a string "Cn", n an integer.
        :param K: Number of Gaussian blobs used to construct the Volume(s).
        :param seed: Random seed for generating random Gaussian blobs.
        :param dtype: dtype for Volume(s)
        """
        super().__init__(L, C, symmetry_type, seed=seed, dtype=dtype)
        self.K = K
        self._check_symmetry()

    def generate(self):
        """
        Generates a LegacyVolume or CnSymmetricVolume that is multiplied by a bump function
        to give compact support within the unit sphere.
        """
        vol = gaussian_blob_vols(
            L=self.L,
            C=self.C,
            symmetry_type=self.symmetry_type,
            seed=self.seed,
            dtype=self.dtype,
        )

        bump_mask = bump_3d(self.L, spread=100, dtype=self.dtype)
        vol = np.multiply(bump_mask, vol)

        return Volume(vol)

    def _check_symmetry(self):
        """
        CompactVolume volumes can have any supported symmetry.
        """


class CnSymmetricVolume(SyntheticVolumeBase):
    """
    A cyclically symmetric Volume constructed with random 3D Gaussian blobs.
    """

    # Note this class can actually inherit everything from LegacyVolume.
    def __init__(self, L, C, symmetry_type, K=16, seed=None, dtype=np.float64):
        """
        :param L: Resolution of the Volume(s) in pixels.
        :param C: Number of Volumes to generate.
        :param symmetry_type: Volume symmetry. A string "Cn", n an integer.
        :param K: Number of Gaussian blobs used to construct the Volume(s).
        :param seed: Random seed for generating random Gaussian blobs.
        :param dtype: dtype for Volume(s)
        """
        super().__init__(L, C, symmetry_type, seed=seed, dtype=dtype)
        self.K = K
        self._check_symmetry()

    def generate(self):
        """
        Called to generate and return an ASPIRE LegacyVolume as a Volume instance.
        """
        return gaussian_blob_vols(
            L=self.L,
            C=self.C,
            symmetry_type=self.symmetry_type,
            seed=self.seed,
            dtype=self.dtype,
        )

    def _check_symmetry(self):
        """
        Checks that `symmetry_type` is Cn.
        """
        if self.symmetry_type is None:
            raise ValueError(
                "Symmetry was not provided. symmetry_type must be 'Cn', n > 1."
            )

        if self.symmetry_type[0].upper() != "C":
            raise ValueError(
                f"Only 'Cn' symmetry supported. Provided symmetry was symmetry_type='{self.symmetry_type}'."
            )

        order = self.symmetry_type[1:] or None
        try:
            order = int(order)
        except Exception:
            raise NotImplementedError(
                f"C{order} symmetry not supported. Only Cn symmetry, where n is an integer, is supported."
            )


def gaussian_blob_vols(L=8, C=2, K=16, symmetry_type=None, seed=None, dtype=np.float64):
    """
    Builds gaussian blob volumes with chosen symmetry type.

    :param L: The resolution of the volume.
    :param C: Number of volumes.
    :param K: The number of gaussian blobs used to generate the volume.
    :param symmetry_type: A string indicating the type of symmetry.
    :param seed: The random seed to produce centers and variances of the gaussian blobs.
    :param dtype: Data type.

    :return: A volume instance from an appropriate volume generator.
    """

    order = 1
    sym_type = None
    if symmetry_type is not None:
        # safer to make string consistent
        symmetry_type = symmetry_type.upper()
        # get the first letter
        sym_type = symmetry_type[0]
        # if there is a number denoting rotational symmetry, get that
        order = symmetry_type[1:] or None

    # map our sym_types to classes of Volumes
    map_sym_to_generator = {
        None: _gaussian_blob_Cn_vols,
        "C": _gaussian_blob_Cn_vols,
        # "D": gaussian_blob_Dn_vols,
        # "T": gaussian_blob_T_vols,
        # "O": gaussian_blob_O_vols,
    }

    sym_types = list(map_sym_to_generator.keys())
    if sym_type not in map_sym_to_generator.keys():
        raise NotImplementedError(
            f"{sym_type} type symmetry is not supported. The following symmetry types are currently supported: {sym_types}."
        )

    try:
        order = int(order)
    except Exception:
        raise NotImplementedError(
            f"{sym_type}{order} symmetry not supported. Only {sym_type}n symmetry, where n is an integer, is supported."
        )

    vols_generator = map_sym_to_generator[sym_type]

    return vols_generator(L=L, C=C, K=K, order=order, seed=seed, dtype=dtype)


def _gaussian_blob_Cn_vols(
    L=8, C=2, K=16, alpha=1, order=1, seed=None, dtype=np.float64
):
    """
    Generate Cn rotationally symmetric volumes composed of Gaussian blobs.
    The volumes are symmetric about the z-axis.

    Defaults to volumes with no symmetry.

    :param L: The size of the volumes
    :param C: The number of volumes to generate
    :param K: The number of blobs each volume is composed of.
    A Cn symmetric volume will be composed of n times K blobs.
    :param order: The order of cyclic symmetry.
    :param alpha: A scale factor of the blob widths

    :return: A Volume instance containing C Gaussian blob volumes with Cn symmetry.
    """

    # Apply symmetry to Q and mu by generating duplicates rotated by symmetry order.
    def _symmetrize_gaussians(Q, D, mu, order):
        angles = np.zeros(shape=(order, 3))
        angles[:, 2] = 2 * np.pi * np.arange(order) / order
        rot = Rotation.from_euler(angles).matrices

        K = Q.shape[0]
        Q_rot = np.zeros(shape=(order * K, 3, 3)).astype(dtype)
        D_sym = np.zeros(shape=(order * K, 3, 3)).astype(dtype)
        mu_rot = np.zeros(shape=(order * K, 3)).astype(dtype)
        idx = 0

        for j in range(order):
            for k in range(K):
                Q_rot[idx] = rot[j].T @ Q[k]
                D_sym[idx] = D[k]
                mu_rot[idx] = rot[j].T @ mu[k]
                idx += 1
        return Q_rot, D_sym, mu_rot

    vols = np.zeros(shape=(C, L, L, L)).astype(dtype)
    with Random(seed):
        for c in range(C):
            Q, D, mu = _gen_gaussians(K, alpha)
            Q_rot, D_sym, mu_rot = _symmetrize_gaussians(Q, D, mu, order)
            vols[c] = _eval_gaussians(L, Q_rot, D_sym, mu_rot, dtype=dtype)
    return Volume(vols)


def _eval_gaussians(L, Q, D, mu, dtype=np.float64):
    """
    Evaluate Gaussian blobs over a 3D grid with centers, mu, orientations, Q, and variances, D.

    :param L: Size of the volume to be populated with Gaussian blobs.
    :param Q: A stack of size (n_blobs) x 3 x 3 of rotation matrices,
        determining the orientation of each blob.
    :param D: A stack of size (n_blobs) x 3 x 3 diagonal matrices,
        whose diagonal entries are the variances of each blob.
    :param mu: An array of size (n_blobs) x 3 containing the centers for each blob.

    :return: An L x L x L array.
    """
    g = grid_3d(L, indexing="xyz", dtype=dtype)
    coords = np.array(
        [g["x"].flatten(), g["y"].flatten(), g["z"].flatten()], dtype=dtype
    )

    n_blobs = Q.shape[0]
    vol = np.zeros(shape=(1, coords.shape[-1])).astype(dtype)

    for k in range(n_blobs):
        coords_k = coords - mu[k, :, np.newaxis]
        coords_k = Q[k].T @ coords_k * np.sqrt(1 / np.diag(D[k, :, :]))[:, np.newaxis]

        vol += np.exp(-0.5 * np.sum(np.abs(coords_k) ** 2, axis=0))

    vol = np.reshape(vol, g["x"].shape)

    return vol


def _gen_gaussians(K, alpha, dtype=np.float64):
    """
    For K gaussians, generate random orientation (Q), mean (mu), and variance (D).

    :param K: Number of gaussians to generate.
    :param alpha: Scalar for peak of gaussians.

    :return: Orientations Q, Variances D, Means mu.
    """
    Q = np.zeros(shape=(K, 3, 3)).astype(dtype)
    D = np.zeros(shape=(K, 3, 3)).astype(dtype)
    mu = np.zeros(shape=(K, 3)).astype(dtype)

    for k in range(K):
        V = randn(3, 3).astype(dtype) / np.sqrt(3)
        Q[k, :, :] = qr(V)[0]
        D[k, :, :] = alpha**2 / 16 * np.diag(np.sum(abs(V) ** 2, axis=0))
        mu[k, :] = 0.5 * randn(3) / np.sqrt(3)

    return Q, D, mu
