import os.path
from unittest import TestCase

import numpy as np

from aspire.utils import (
    fix_signs,
    im_to_vec,
    mat_to_vec,
    roll_dim,
    symmat_to_vec_iso,
    unroll_dim,
    vec_to_im,
    vec_to_symmat,
    vec_to_symmat_iso,
    vec_to_vol,
    vecmat_to_volmat,
    vol_to_vec,
    volmat_to_vecmat,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


class MatrixTestCase(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testUnrollDims(self):
        m = np.arange(1, 1201).reshape((5, 2, 10, 3, 4), order="F")
        m2, sz = unroll_dim(
            m, 2
        )  # second argument is 1-indexed - all dims including and after this are unrolled

        # m2 will now have shape (5, (2x10x3x4)) = (5, 240)
        self.assertEqual(m2.shape, (5, 240))
        # The values should still be filled in with the first axis values changing fastest
        self.assertTrue(np.allclose(m2[:, 0], np.array([1, 2, 3, 4, 5])))

        # sz are the dimensions that were unrolled
        self.assertEqual(sz, (2, 10, 3, 4))

    def testRollDims(self):
        m = np.arange(1, 1201).reshape((5, 2, 120), order="F")
        m2 = roll_dim(m, (10, 3, 4))

        # m2 will now have shape (5, 2, 10, 3, 4)
        self.assertEqual(m2.shape, (5, 2, 10, 3, 4))
        # The values should still be filled in with the first axis values changing fastest
        self.assertTrue(np.allclose(m2[:, 0, 0, 0, 0], np.array([1, 2, 3, 4, 5])))

    def testImToVec1(self):
        m = np.empty((3, 3, 10))
        m2 = im_to_vec(m)

        self.assertEqual(m2.shape, (9, 10))

    def testImToVec2(self):
        m = np.empty((3, 3))
        m2 = im_to_vec(m)

        self.assertEqual(m2.shape, (9,))

    def testVecToIm1(self):
        m = np.empty((25, 10))
        m2 = vec_to_im(m)

        self.assertEqual(m2.shape, (5, 5, 10))

    def testVecToIm2(self):
        m = np.empty((16,))
        m2 = vec_to_im(m)

        self.assertEqual(m2.shape, (4, 4))

    def testVolToVec1(self):
        m = np.empty((3, 3, 3, 10))
        m2 = vol_to_vec(m)

        self.assertEqual(m2.shape, (27, 10))

    def testVolToVec2(self):
        m = np.empty((3, 3, 3))
        m2 = vol_to_vec(m)

        self.assertEqual(m2.shape, (27,))

    def testVecToVol1(self):
        m = np.empty((27, 10))
        m2 = vec_to_vol(m)

        self.assertEqual(m2.shape, (3, 3, 3, 10))

    def testVecToVol2(self):
        m = np.empty((27,))
        m2 = vec_to_vol(m)

        self.assertEqual(m2.shape, (3, 3, 3))

    def testVecmatToVolmat(self):
        m = np.empty((8, 27, 10))
        m2 = vecmat_to_volmat(m)

        self.assertEqual(m2.shape, (2, 2, 2, 3, 3, 3, 10))

    def testVolmatToVecmat(self):
        m = np.empty((3, 3, 3, 2, 2, 2, 5))
        m2 = volmat_to_vecmat(m)

        self.assertEqual(m2.shape, (27, 8, 5))

    def testMatToVec1(self):
        m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        v = mat_to_vec(m)
        self.assertTrue(np.allclose(v, np.array([1, 4, 7, 2, 5, 8, 3, 6, 9])))

    def testMatToVec2(self):
        m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        # Make 2 copies depthwise
        m = np.dstack((m, m))

        v = mat_to_vec(m)
        self.assertTrue(
            np.allclose(
                v,
                np.array(
                    [
                        [1, 1],
                        [4, 4],
                        [7, 7],
                        [2, 2],
                        [5, 5],
                        [8, 8],
                        [3, 3],
                        [6, 6],
                        [9, 9],
                    ]
                ),
            )
        )

    def testMatToVecSymm1(self):
        # We create an unsymmetric matrix and pass it to the functions as a symmetric matrix,
        # just so we can closely inspect the returned values without confusion
        m = np.array([[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]])
        v = mat_to_vec(m, is_symmat=True)
        # Notice the order of the elements in symmetric matrix - axis 0 first, then axis 1
        self.assertTrue(np.allclose(v, np.array([0, 1, 2, 3, 5, 6, 7, 10, 11, 15])))

    def testMatToVecSymm2(self):
        # We create an unsymmetric matrix and pass it to the functions as a symmetric matrix,
        # just so we can closely inspect the returned values without confusion
        m = np.array([[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]])
        # Make 2 copies depthwise
        m = np.dstack((m, m))

        v = mat_to_vec(m, is_symmat=True)
        # Notice the order of the elements in symmetric matrix - axis 0 first, then axis 1
        self.assertTrue(
            np.allclose(
                v,
                np.array(
                    [
                        [0, 0],
                        [1, 1],
                        [2, 2],
                        [3, 3],
                        [5, 5],
                        [6, 6],
                        [7, 7],
                        [10, 10],
                        [11, 11],
                        [15, 15],
                    ]
                ),
            )
        )

    def testMatToVecSymmIso(self):
        # Very similar to the case above, except that the resulting matrix is reweighted.

        # We create an unsymmetric matrix and pass it to the functions as a symmetric matrix,
        # just so we can closely inspect the returned values without confusion
        m = np.array(
            [[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]],
            dtype=np.float32,
        )

        # Make 2 copies depthwise
        m = np.dstack((m, m))

        v = symmat_to_vec_iso(m)
        # Notice the order of the elements in symmetric matrix - axis 0 first, then axis 1
        self.assertTrue(
            np.allclose(
                v,
                np.array(
                    [
                        [0, 0],
                        [1.4142, 1.4142],
                        [2.8284, 2.8284],
                        [4.2426, 4.2426],
                        [5, 5],
                        [8.4853, 8.4853],
                        [9.8995, 9.8995],
                        [10, 10],
                        [15.5563, 15.5563],
                        [15, 15],
                    ]
                ),
            )
        )

    def testVecToMatSymm1(self):
        v = np.array(
            [
                [0, 0],
                [1, 1],
                [2, 2],
                [3, 3],
                [5, 5],
                [6, 6],
                [7, 7],
                [10, 10],
                [11, 11],
                [15, 15],
            ]
        )

        m = vec_to_symmat(v)
        self.assertTrue(
            np.allclose(
                m[:, :, 0],
                np.array([[0, 1, 2, 3], [1, 5, 6, 7], [2, 6, 10, 11], [3, 7, 11, 15]]),
            )
        )
        self.assertTrue(
            np.allclose(
                m[:, :, 1],
                np.array([[0, 1, 2, 3], [1, 5, 6, 7], [2, 6, 10, 11], [3, 7, 11, 15]]),
            )
        )

    def testVecToMatSymm2(self):
        v = np.array([0, 1, 2, 3, 5, 6, 7, 10, 11, 15])

        m = vec_to_symmat(v)
        self.assertTrue(
            np.allclose(
                m[:, :],
                np.array([[0, 1, 2, 3], [1, 5, 6, 7], [2, 6, 10, 11], [3, 7, 11, 15]]),
            )
        )

    def testVecToMatSymmIso(self):
        # Very similar to the case above, except that the resulting matrix is reweighted.
        v = np.array(
            [
                [0, 0],
                [1, 1],
                [2, 2],
                [3, 3],
                [5, 5],
                [6, 6],
                [7, 7],
                [10, 10],
                [11, 11],
                [15, 15],
            ],
            dtype=np.float32,
        )

        m = vec_to_symmat_iso(v)
        self.assertTrue(
            np.allclose(
                m[:, :, 0],
                np.array(
                    [
                        [0, 0.70710678, 1.41421356, 2.12132034],
                        [0.70710678, 5, 4.24264069, 4.94974747],
                        [1.41421356, 4.24264069, 10, 7.77817459],
                        [2.12132034, 4.94974747, 7.77817459, 15],
                    ]
                ),
            )
        )
        self.assertTrue(
            np.allclose(
                m[:, :, 1],
                np.array(
                    [
                        [0, 0.70710678, 1.41421356, 2.12132034],
                        [0.70710678, 5, 4.24264069, 4.94974747],
                        [1.41421356, 4.24264069, 10, 7.77817459],
                        [2.12132034, 4.94974747, 7.77817459, 15],
                    ]
                ),
            )
        )

    def testFixSigns(self):
        """
        Tests `fix_signs` util function.
        """

        # Create simple array
        x = np.arange(25).reshape(5, 5)
        # Set diagonal elements = -1
        x[np.diag_indices_from(x)] *= -1
        # Negate largest elem (last row) of first col
        x[-1, 0] *= -1

        # Now we expect fix_signs to negate the first and last column,
        #  otherwise should be identical.
        y = x.copy()
        y[:, (0, -1)] *= -1
        self.assertTrue(np.allclose(fix_signs(x), y))

        # Should work for complex cases too.
        x = x + x * 1j
        y = x.copy()
        y[:, (0, -1)] *= -1
        self.assertTrue(np.allclose(fix_signs(x), y))

        # Insert a zero column to spice things up
        x[:, 3] = 0
        y[:, 3] = 0
        self.assertTrue(np.allclose(fix_signs(x), y))
