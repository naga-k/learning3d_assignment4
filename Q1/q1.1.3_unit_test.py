import torch
from model import Scene, Gaussians

def test_get_idxs_to_filter_and_sort():
    # Create test depth values
    z_vals = torch.tensor([-1.0, 5.0, 2.0, -2.0, 3.0, 0.0])
    
    # Create dummy Gaussians object and Scene
    gaussians = Gaussians("random", "cpu", num_points=len(z_vals))
    scene = Scene(gaussians)
    
    # Get sorted and filtered indices
    idxs = scene.get_idxs_to_filter_and_sort(z_vals)
    
    # Expected indices after filtering negative values and sorting
    expected_idxs = torch.tensor([5, 2, 4, 1])
    
    # Test if output matches expected
    assert torch.allclose(idxs, expected_idxs), f"Expected {expected_idxs}, but got {idxs}"
    
    # Test if returned indices give sorted positive depths
    sorted_z_vals = z_vals[idxs]
    assert torch.all(sorted_z_vals >= 0), "Not all returned depths are positive"
    assert torch.all(sorted_z_vals[:-1] <= sorted_z_vals[1:]), "Depths are not sorted"
    
    print("All tests passed!")

if __name__ == "__main__":
    test_get_idxs_to_filter_and_sort()