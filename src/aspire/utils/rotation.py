"""
Define a Rotation Class for customized rotation operations used by ASPIRE.
"""

import numpy as np
from numpy.linalg import norm
from scipy.linalg import svd
from scipy.spatial.transform import Rotation as sp_rot

from aspire.utils.random import Random


class Rotation:
    def __init__(self, matrices):
        """
         Initialize a Rotation object

        :param matrices: Rotation matrices to initialize Rotation object.
        """
        if matrices.ndim == 2:
            matrices = matrices.reshape((1, 3, 3))
        assert matrices.ndim == 3 and matrices.shape[-2:] == (
            3,
            3,
        ), f"Bad rotation matrix shape: {matrices.shape}"
        self._matrices = matrices
        self._seq_order = "ZYZ"

    def __str__(self):
        """
        String representation.
        """
        return f"Rotation stack consisting of {len(self)} elements of {self.dtype} type"

    def __len__(self):
        return self._matrices.shape[0]

    def __getitem__(self, item):
        return self._matrices[item]

    def __setitem__(self, key, value):
        self._matrices[key] = value

    def __mul__(self, other):
        output = self._matrices @ other.matrices
        return Rotation(output)

    @property
    def dtype(self):
        """
        :return: dtype of Rotation matrices
        """
        return self._matrices.dtype

    @property
    def matrices(self):
        """
        :return: Rotation matrices as a n x 3 x 3 array
        """
        return self._matrices

    @property
    def angles(self):
        """
        :return: Rotation matrices as a n x 3 array
        """
        rotations = sp_rot.from_matrix(self._matrices.astype(self.dtype))
        return rotations.as_euler(self._seq_order, degrees=False).astype(self.dtype)

    def invert(self):
        """
        Apply transpose operation to all rotation matrices

        :return: The set of transposed matrices
        """
        return Rotation(
            matrices=np.transpose(self._matrices, axes=(0, 2, 1)),
        )

    def find_registration(self, rots_ref):
        """
        Register estimated orientations to reference ones.

        Finds the orthogonal transformation that best aligns the estimated rotations
        to the reference rotations.

        :param rots_ref: The reference Rotation object to which we would like to align
            with data matrices in the form of a n-by-3-by-3 array.
        :return: o_mat, optimal orthogonal 3x3 matrix to align the two sets;
                flag, flag==1 then J conjugacy is required and 0 is not.
        """
        rots = self._matrices
        rots_ref = rots_ref.matrices.astype(self.dtype)
        assert (
            rots.shape == rots_ref.shape
        ), "Two sets of rotations must have same dimensions."
        K = rots.shape[0]

        # Reflection matrix
        J = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])

        Q1 = np.zeros((3, 3), dtype=rots.dtype)
        Q2 = np.zeros((3, 3), dtype=rots.dtype)

        for k in range(K):
            R = rots[k, :, :]
            Rref = rots_ref[k, :, :]
            Q1 = Q1 + R @ Rref.T
            Q2 = Q2 + (J @ R @ J) @ Rref.T

        # Compute the two possible orthogonal matrices which register the
        # estimated rotations to the true ones.
        Q1 = Q1 / K
        Q2 = Q2 / K

        # We are registering one set of rotations (the estimated ones) to
        # another set of rotations (the true ones). Thus, the transformation
        # matrix between the two sets of rotations should be orthogonal. This
        # matrix is either Q1 if we recover the non-reflected solution, or Q2,
        # if we got the reflected one. In any case, one of them should be
        # orthogonal.

        err1 = norm(Q1 @ Q1.T - np.eye(3), ord="fro")
        err2 = norm(Q2 @ Q2.T - np.eye(3), ord="fro")

        # In any case, enforce the registering matrix O to be a rotation.
        if err1 < err2:
            # Use Q1 as the registering matrix
            U, _, V = svd(Q1)
            flag = 0
        else:
            # Use Q2 as the registering matrix
            U, _, V = svd(Q2)
            flag = 1

        Q_mat = U @ V

        return Q_mat, flag

    def apply_registration(self, Q_mat, flag):
        """
        Get aligned Rotation object to reference ones.

        Calculated aligned rotation matrices from the orthogonal transformation
        that best aligns the estimated rotations to the reference rotations.

        :param Q_mat:  optimal orthogonal 3x3 transformation matrix
        :param flag:  flag==1 then J conjugacy is required and 0 is not
        :return: regrot, aligned Rotation object
        """
        rots = self._matrices
        K = rots.shape[0]

        # Reflection matrix
        J = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])

        regrot = np.zeros_like(rots)
        for k in range(K):
            R = rots[k, :, :]
            if flag == 1:
                R = J @ R @ J
            regrot[k, :, :] = Q_mat.T @ R
        aligned_rots = Rotation(regrot)
        return aligned_rots

    def register(self, rots_ref):
        """
         Estimate global orientation and return an aligned Rotation object.

        :param rots_ref: The reference Rotation object to which we would like
            to align with data matrices in the form of a n-by-3-by-3 array.
        :return: an aligned Rotation object
        """
        Q_mat, flag = self.find_registration(rots_ref)
        return self.apply_registration(Q_mat, flag)

    def mse(self, rots_ref):
        """
        Calculate MSE between the estimated orientations to reference ones.

        :param rots_reg: The estimated Rotation object after alignment
             with data matrices in the form of a n-by-3-by-3 array.
        :param rots_ref: The reference Rotation object.
        :return: The MSE value between two sets of rotations.
        """
        aligned_rots = self.register(rots_ref)
        rots_reg = aligned_rots.matrices
        rots_ref = rots_ref.matrices
        assert (
            rots_reg.shape == rots_ref.shape
        ), "Two sets of rotations must have same dimensions."
        K = rots_reg.shape[0]

        diff = np.zeros(K)
        mse = 0
        for k in range(K):
            diff[k] = norm(rots_reg[k, :, :] - rots_ref[k, :, :], ord="fro")
            mse += diff[k] ** 2
        mse = mse / K
        return mse

    def common_lines(self, i, j, ell):
        """
        Compute the common line induced by rotation matrices i and j.

        :param i: The index of first rotation matrix of 3-by-3 array.
        :param j: The index of second rotation matrix of 3-by-3 array.
        :param ell: The total number of common lines.
        :return: The common line indices for both first and second rotations.
        """
        r1 = self._matrices[i]
        r2 = self._matrices[j]
        ut = np.dot(r2, r1.T)
        alpha_ij = np.arctan2(ut[2, 0], -ut[2, 1]) + np.pi
        alpha_ji = np.arctan2(ut[0, 2], -ut[1, 2]) + np.pi

        ell_ij = alpha_ij * ell / (2 * np.pi)
        ell_ji = alpha_ji * ell / (2 * np.pi)

        ell_ij = int(np.mod(np.round(ell_ij), ell))
        ell_ji = int(np.mod(np.round(ell_ji), ell))

        return ell_ij, ell_ji

    @staticmethod
    def from_euler(values, dtype=np.float32):
        """
        build rotation object from Euler angles in radians

        :param dtype:  data type for rotational angles and matrices
        :param values: Rotation angles in radians, as a n x 3 array
        :return: new Rotation object
        """
        rotations = sp_rot.from_euler("ZYZ", values, degrees=False)
        matrices = rotations.as_matrix().astype(dtype)
        return Rotation(matrices)

    @staticmethod
    def about_axis(axis, angles, dtype=np.float32):
        """
        Build rotation object from axis and angles of rotation.

        :param axis: A string denoting the axis of rotation. "x", "y", or "z".
        :param angles: Rotation angles in radians. `angles` can be a single value,
            or an array of shape (N,) or (N,1).
        :param dtype: Data type for rotation matrices.

        :return: Rotation object
        """
        axes = ["x", "y", "z"]
        if axis.lower() not in axes:
            raise ValueError("`axis` must be 'x', 'y', or 'z'.")

        angles = np.asarray(angles, dtype=dtype)
        if angles.ndim > 2:
            raise ValueError(
                f"`angles` must be float, 1D array, or 2D array. Got shape {angles.shape}."
            )
        elif angles.ndim == 2 and angles.shape[-1] != 1:
            raise ValueError(
                f"Expected `angles` to have shape (N,1), got {angles.shape}"
            )

        rotation = sp_rot.from_euler(axis, angles, degrees=False)
        matrix = rotation.as_matrix().astype(dtype)
        return Rotation(matrix)

    @staticmethod
    def from_rotvec(vec, dtype=np.float32):
        """
        Build a Rotation object from rotation vectors. A rotation vector is a
        3D vector which is co-directional to the axis of rotation and whose norm
        gives the angle of rotation in radians. The angle of rotation is counter-clockwise
        about the axis.

        :param vec: array_like, shape (N, 3) or (3,)
        :return: Rotation object
        """
        rots = sp_rot.from_rotvec(vec)
        matrices = rots.as_matrix().astype(dtype)
        return Rotation(matrices)

    @staticmethod
    def from_matrix(values, dtype=np.float32):
        """
        build rotation object from rotational matrices

        :param dtype:  data type for rotational angles and matrices
        :param values: Rotation matrices, as a n x 3 x 3 array
        :return: new Rotation object
        """
        return Rotation(values.astype(dtype))

    @staticmethod
    def generate_random_rotations(
        n,
        seed=None,
        dtype=np.float32,
    ):
        """
        Generate Rotation object with random 3D rotation matrices

        :param n: The number of rotation matrices to generate
        :param seed: Random integer seed to use. If None,
            the current random state is used.
        :param dtype:  data type for rotational angles and matrices
        :return: A new Rotation object
        """
        # Generate random rotation angles, in radians
        with Random(seed):
            angles = np.column_stack(
                (
                    np.random.random(n) * 2 * np.pi,
                    np.arccos(2 * np.random.random(n) - 1),
                    np.random.random(n) * 2 * np.pi,
                )
            ).astype(dtype=dtype)

        return Rotation.from_euler(angles, dtype=dtype)

    @staticmethod
    def angle_dist(r1, r2, dtype=np.float32):
        """
        Find the angular distance between two rotation matrices. We first compute
        the rotation between the two rotation matrices, r = r1 @ r2.T. Then using
        the axis-angle representation of r we find the angle between r1 and r2.

        :param r1: A 3x3 rotation matrix
        :paran r2: A 3x3 rotation matrix

        :return: The angular distance between r1 and r2 in radians.
        """
        r = r1 @ r2.T
        tr_r = np.trace(r, dtype=dtype)
        if np.allclose(tr_r, 3):
            dist = 0
        else:
            theta = (tr_r - 1) / 2
            theta = max(min(theta, 1), -1)  # Clamp theta in [-1,1]
            dist = np.arccos(theta, dtype=dtype)
        return dist
