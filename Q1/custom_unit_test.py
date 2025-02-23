import torch
from model import Gaussians
import unittest
from pytorch3d.transforms import quaternion_to_matrix

class TestGaussians(unittest.TestCase):
    def test_compute_cov_3D_anisotropic(self):
        # Initialize test data
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gaussians = Gaussians(init_type="random", device=device, num_points=3, isotropic=False)
        
        # Create test quaternions representing identity rotation
        quats = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],  # Identity quaternion
            [0.707, 0.707, 0.0, 0.0],  # 90-degree rotation around x
            [0.707, 0.0, 0.707, 0.0],  # 90-degree rotation around y
        ], device=device)
        
        # Create test scales
        scales = torch.tensor([
            [1.0, 2.0, 3.0],
            [0.5, 1.0, 1.5],
            [2.0, 1.0, 0.5]
        ], device=device)
        
        # Compute actual covariance
        cov_3D = gaussians.compute_cov_3D(quats, scales)
        
        # Compute expected covariance manually
        expected_covs = []
        for i in range(3):
            R = quaternion_to_matrix(quats[i:i+1])  # Convert to 3x3 rotation matrix
            S = torch.diag(scales[i] * scales[i])  # Create diagonal scale matrix
            expected_cov = R @ S @ R.transpose(-2, -1)

            expected_covs.append(expected_cov)
        
        expected_cov_3D = torch.concat(expected_covs, dim = 0)
        
        # Test if results match
        torch.testing.assert_close(cov_3D, expected_cov_3D, rtol=1e-4, atol=1e-4)
        
        # Test output shape
        self.assertEqual(cov_3D.shape, (3, 3, 3))
        
        # Test if covariance matrices are symmetric
        torch.testing.assert_close(
            cov_3D, cov_3D.transpose(-2, -1), 
            rtol=1e-4, atol=1e-4, 
            msg="Covariance matrices should be symmetric"
        )

if __name__ == '__main__':
    unittest.main()