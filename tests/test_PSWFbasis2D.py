import os.path
from unittest import TestCase

import numpy as np

from aspire.basis import PSWFBasis2D
from aspire.image import Image

from ._basis_util import UniversalBasisMixin

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


class PSWFBasis2DTestCase(TestCase, UniversalBasisMixin):
    def setUp(self):
        self.basis = PSWFBasis2D((8, 8), 1.0, 1.0)
        self.L = 8
        self.dtype = np.float64
        self.basis = PSWFBasis2D((self.L, self.L), 1.0, 1.0, dtype=self.dtype)

    def tearDown(self):
        pass

    def testPSWFBasis2DEvaluate_t(self):
        img_ary = np.load(
            os.path.join(DATA_DIR, "ffbbasis2d_xcoeff_in_8_8.npy")
        ).T  # RCOPT
        images = Image(img_ary)

        result = self.basis.evaluate_t(images)

        coeffs = np.load(
            os.path.join(DATA_DIR, "pswf2d_vcoeffs_out_8_8.npy")
        ).T  # RCOPT

        # make sure both real and imaginary parts are consistent.
        self.assertTrue(
            np.allclose(np.real(result), np.real(coeffs))
            and np.allclose(np.imag(result) * 1j, np.imag(coeffs) * 1j)
        )

    def testPSWFBasis2DEvaluate(self):
        coeffs = np.load(
            os.path.join(DATA_DIR, "pswf2d_vcoeffs_out_8_8.npy")
        ).T  # RCOPT
        result = self.basis.evaluate(coeffs)
        images = np.load(os.path.join(DATA_DIR, "pswf2d_xcoeff_out_8_8.npy")).T  # RCOPT
        self.assertTrue(np.allclose(result.asnumpy(), images))
