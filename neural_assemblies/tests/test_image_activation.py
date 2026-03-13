# test_image_activation.py

import unittest
import numpy as np

from brain import Brain


class TestImageActivation(unittest.TestCase):
    def setUp(self):
        self.brain = Brain(p=0.1, seed=42)
        self.brain.add_area("IMG", n=32, k=5, beta=0.05, explicit=True)

    def test_image_smaller_than_n_pads(self):
        img = np.arange(10).astype(np.float32)
        self.brain.activate_with_image("IMG", img)
        area = self.brain.area_by_name["IMG"]
        self.assertEqual(len(area.winners), area.k)
        self.assertTrue(np.all(area.winners < area.n))

    def test_image_larger_than_n_crops(self):
        img = np.arange(100).astype(np.float32)
        self.brain.activate_with_image("IMG", img)
        area = self.brain.area_by_name["IMG"]
        self.assertEqual(len(area.winners), area.k)
        self.assertTrue(np.all(area.winners < area.n))

    def test_constant_image_normalization_stable(self):
        img = np.ones(20, dtype=np.float32)
        # Should not error due to ptp=0; we add epsilon in normalization path
        self.brain.activate_with_image("IMG", img)
        area = self.brain.area_by_name["IMG"]
        self.assertEqual(len(area.winners), area.k)

    def test_winners_are_topk_after_normalization(self):
        # Construct an image with strictly increasing values
        n = self.brain.area_by_name["IMG"].n
        k = self.brain.area_by_name["IMG"].k
        img = np.arange(n, dtype=np.float32)
        self.brain.activate_with_image("IMG", img)
        area = self.brain.area_by_name["IMG"]
        expected_topk = np.argsort(-img)[:k]
        # Normalization is monotonic; top-k preserved
        np.testing.assert_array_equal(np.sort(area.winners), np.sort(expected_topk))

    def test_torch_tensor_input_supported(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("torch not available")
        import torch as _torch
        n = self.brain.area_by_name["IMG"].n
        img_t = _torch.arange(n, dtype=_torch.float32)
        self.brain.activate_with_image("IMG", img_t)
        area = self.brain.area_by_name["IMG"]
        self.assertEqual(len(area.winners), area.k)
        self.assertTrue(np.all(area.winners < n))


if __name__ == "__main__":
    unittest.main()


