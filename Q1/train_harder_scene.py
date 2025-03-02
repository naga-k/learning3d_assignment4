import os
import torch
import imageio
import argparse
import numpy as np

from PIL import Image
from tqdm import tqdm
from model import Scene, Gaussians
from torch.utils.data import DataLoader
from data_utils import visualize_renders
from data_utils_harder_scene import get_nerf_datasets, trivial_collate

from pytorch3d.renderer import PerspectiveCameras
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from torch.nn import L1Loss
from pytorch_msssim import ssim

def make_trainable(gaussians):

    ### YOUR CODE HERE ###
    # HINT: You can access and modify parameters from gaussians
    gaussians.pre_act_opacities.requires_grad = True
    gaussians.pre_act_scales.requires_grad = True
    gaussians.colours.requires_grad = True
    gaussians.means.requires_grad = True

def setup_optimizer_scheduler(gaussians):

    gaussians.check_if_trainable()

    ### YOUR CODE HERE ###
    # HINT: Modify the learning rates to reasonable values. We have intentionally
    # set very high learning rates for all parameters.
    # HINT: Consider reducing the learning rates for parameters that seem to vary too
    # fast with the default settings.
    # HINT: Consider setting different learning rates for different sets of parameters.
    parameters = [
        {'params': [gaussians.pre_act_opacities], 'lr': 0.01, "name": "opacities"},
        {'params': [gaussians.pre_act_scales], 'lr': 0.005, "name": "scales"},
        {'params': [gaussians.colours], 'lr': 0.02, "name": "colours"},
        {'params': [gaussians.means], 'lr': 0.01, "name": "means"},
    ]
    optimizer = torch.optim.Adam(parameters, lr=0.0, eps=1e-15)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)


    return optimizer, scheduler

def densify_gaussians(gaussians, grad_threshold=0.01, max_points=20000):
    """
    Add new Gaussians where gradient is high (areas that need more detail)
    Returns: Number of points added
    """
    if gaussians.means.grad is None:
        return 0
    
    # Calculate gradient norm for each mean position
    mean_grad_norm = torch.norm(gaussians.means.grad, dim=1)
    
    # Find points with high gradients (needing more detail)
    high_grad_indices = torch.where(mean_grad_norm > grad_threshold)[0]
    
    # Limit the number of points to densify to avoid excessive growth
    if len(high_grad_indices) > 500:
        # Select a subset if too many points qualify
        perm = torch.randperm(len(high_grad_indices))
        high_grad_indices = high_grad_indices[perm[:500]]
    
    # Skip if no points to densify or if we'd exceed max points
    if len(high_grad_indices) == 0 or len(gaussians.means) + len(high_grad_indices)*2 > max_points:
        return 0
    
    # Create new Gaussians from high gradient points
    new_means, new_pre_act_scales, new_pre_act_opacities, new_colours, new_pre_act_quats = (
        gaussians.create_new_gaussians_from_existing(high_grad_indices)
    )
    
    # Delete originals that were split
    gaussians.remove(high_grad_indices)
    
    # Add the new Gaussians
    gaussians.add(new_means, new_pre_act_scales, new_pre_act_opacities, new_colours, new_pre_act_quats)
    
    return len(high_grad_indices) * 2 - len(high_grad_indices)  # Net points added

def prune_gaussians(gaussians, opacity_threshold=0.01, scale_threshold=0.001):
    """
    Remove Gaussians that contribute little to the rendering
    Returns: Number of points removed
    """
    # Apply sigmoid to get actual opacity values
    opacities = torch.sigmoid(gaussians.pre_act_opacities)
    
    # Calculate scales from pre-activated scales
    scales = torch.exp(gaussians.pre_act_scales)
    
    # For anisotropic Gaussians, compute the norm of the scale
    if not gaussians.is_isotropic:
        scales = torch.norm(scales, dim=1)
    else:
        scales = scales.squeeze()
    
    # Find points with low opacity or small scale
    indices_to_remove = torch.where(
        (opacities < opacity_threshold) | (scales < scale_threshold)
    )[0]
    
    # Skip if no points to prune
    if len(indices_to_remove) == 0:
        return 0
    
    # Limit pruning to avoid removing too many points at once
    if len(indices_to_remove) > len(gaussians.means) // 2:
        # Don't remove more than half the points at once
        perm = torch.randperm(len(indices_to_remove))
        indices_to_remove = indices_to_remove[perm[:len(gaussians.means) // 2]]
    
    # Remove low-contribution points
    gaussians.remove(indices_to_remove)
    
    return len(indices_to_remove)

def ndc_to_screen_camera(camera, img_size = (128, 128)):

    min_size = min(img_size[0], img_size[1])

    screen_focal = camera.focal_length * min_size / 2.0
    screen_principal = torch.tensor([[img_size[0]/2, img_size[1]/2]]).to(torch.float32)

    return PerspectiveCameras(
        R=camera.R, T=camera.T, in_ndc=False,
        focal_length=screen_focal, principal_point=screen_principal,
        image_size=(img_size,),
    )

def run_training(args):

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path, exist_ok=True)

    train_dataset, val_dataset, _ = get_nerf_datasets(
        dataset_name="materials", data_root=args.data_path,
        image_size=[128, 128],
    )

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=0,
        drop_last=True, collate_fn=trivial_collate
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=0,
        drop_last=True, collate_fn=trivial_collate
    )
    train_itr = iter(train_loader)

    # Preparing some code for visualization
    viz_gif_path_1 = os.path.join(args.out_path, "q1_harder_training_progress.gif")
    viz_gif_path_2 = os.path.join(args.out_path, "q1_harder_training_final_renders.gif")
    viz_idxs = np.linspace(0, len(train_dataset)-1, 5).astype(np.int32)[:4]

    gt_viz_imgs = [(train_dataset[i]["image"]*255.0).numpy().astype(np.uint8) for i in viz_idxs]
    gt_viz_imgs = [np.array(Image.fromarray(x).resize((256, 256))) for x in gt_viz_imgs]
    gt_viz_img = np.concatenate(gt_viz_imgs, axis=1)

    viz_cameras = [ndc_to_screen_camera(train_dataset[i]["camera"]).cuda() for i in viz_idxs]

    # Init gaussians and scene
    gaussians = Gaussians(
        num_points=10000, init_type="random",
        device=args.device, isotropic=True
    )
    scene = Scene(gaussians)

    # Making gaussians trainable and setting up optimizer
    make_trainable(gaussians)
    optimizer, scheduler = setup_optimizer_scheduler(gaussians)
    l1_loss = L1Loss()

    # Add configuration for densification and pruning
    densification_interval = 10  # Perform densification every 100 iterations
    pruning_interval = 20        # Perform pruning every 200 iterations
    max_points = 25000            # Maximum number of Gaussian points

    # Training loop
    viz_frames = []
    for itr in range(args.num_itrs):

        # Fetching data
        try:
            data = next(train_itr)
        except StopIteration:
            train_itr = iter(train_loader)
            data = next(train_itr)

        gt_img = data[0]["image"].cuda()
        camera = ndc_to_screen_camera(data[0]["camera"]).cuda()

        # Rendering scene using gaussian splatting
        ### YOUR CODE HERE ###
        # HINT: Can any function from the Scene class help?
        # HINT: Set bg_colour to (0.0, 0.0, 0.0)
        # HINT: Set img_size to (128, 128)
        # HINT: Get per_splat from args.gaussians_per_splat
        # HINT: camera is available above
        pred_img, depth, mask = scene.render(camera=camera, per_splat=args.gaussians_per_splat, img_size=(gt_img.shape[0], gt_img.shape[1]))

        # Compute loss
        ### YOUR CODE HERE ###
        l1 = l1_loss(pred_img, gt_img)
        # Permute and add batch dimension to match required format [batch, channels, height, width]
        pred_img_ssim = pred_img.permute(2, 0, 1).unsqueeze(0)  # Change from [H,W,C] to [1,C,H,W]
        gt_img_ssim = gt_img.permute(2, 0, 1).unsqueeze(0)      # Change from [H,W,C] to [1,C,H,W]
        d_ssim = 1 - ssim(pred_img_ssim, gt_img_ssim)
        
        lambda_val = 0.2
        loss = (1 - lambda_val) * l1 + lambda_val * d_ssim

        loss.backward()
        optimizer.step()
        scheduler.step()

        print(f"[*] Itr: {itr:07d} | Loss: {loss:0.3f}")

        # Run pruning (remove low-contribution gaussians)
        if itr > 0 and itr % pruning_interval == 0:
            num_removed = prune_gaussians(gaussians, 
                                         opacity_threshold=0.005, 
                                         scale_threshold=0.001)
            if num_removed > 0:
                print(f"[*] Pruning: Removed {num_removed} low-contribution Gaussians")
                # Need to update the optimizer since parameters changed
                # First detach all tensors from the old optimizer
                optimizer.zero_grad()
                # Now recreate the optimizer with the new parameters
                make_trainable(gaussians)  # Reapply requires_grad=True
                optimizer, scheduler = setup_optimizer_scheduler(gaussians)

        # Run densification (add new gaussians in important areas)
        if itr > 0 and itr % densification_interval == 0:
            num_added = densify_gaussians(gaussians, 
                                         grad_threshold=0.0002, 
                                         max_points=max_points)
            if num_added > 0:
                print(f"[*] Densification: Added {num_added} new Gaussians")
                # Need to update the optimizer since parameters changed
                # First detach all tensors from the old optimizer
                optimizer.zero_grad()
                # Now recreate the optimizer with the new parameters
                make_trainable(gaussians)  # Reapply requires_grad=True
                optimizer, scheduler = setup_optimizer_scheduler(gaussians)

        if itr % args.viz_freq == 0:
            viz_frame = visualize_renders(
                scene, gt_viz_img,
                viz_cameras, (128, 128)
            )
            viz_frames.append(viz_frame)

        if itr % 100 == 0:
            print(f"[*] Current number of Gaussians: {len(gaussians)}")

        optimizer.zero_grad()

    print("[*] Training Completed.")

    # Saving training progess GIF
    imageio.mimwrite(viz_gif_path_1, viz_frames, loop=0, duration=(1/10.0)*1000)

    # Creating renderings of the training views after training is completed.
    frames = []
    viz_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=False, num_workers=0,
        drop_last=True, collate_fn=trivial_collate
    )
    for viz_data in tqdm(viz_loader, desc="Creating Visualization"):
        gt_img = viz_data[0]["image"].cuda()
        camera = ndc_to_screen_camera(viz_data[0]["camera"]).cuda()

        with torch.no_grad():

            # Rendering scene using gaussian splatting
            ### YOUR CODE HERE ###
            # HINT: Can any function from the Scene class help?
            # HINT: Set bg_colour to (0.0, 0.0, 0.0)
            # HINT: Set img_size to (128, 128)
            # HINT: Get per_splat from args.gaussians_per_splat
            # HINT: camera is available above
            pred_img, depth, mask = scene.render(camera=camera, per_splat=args.gaussians_per_splat, img_size=(gt_img.shape[0], gt_img.shape[1]))

        pred_npy = pred_img.detach().cpu().numpy()
        pred_npy = (np.clip(pred_npy, 0.0, 1.0) * 255.0).astype(np.uint8)
        frames.append(pred_npy)

    # Saving renderings
    imageio.mimwrite(viz_gif_path_2, frames, loop=0, duration=(1/10.0)*1000)

    # Running evaluation using the test dataset
    psnr_vals, ssim_vals = [], []
    for val_data in tqdm(val_loader, desc="Running Evaluation"):

        gt_img = val_data[0]["image"].cuda()
        camera = ndc_to_screen_camera(val_data[0]["camera"]).cuda()

        with torch.no_grad():

            # Rendering scene using gaussian splatting
            # Rendering scene using gaussian splatting
            ### YOUR CODE HERE ###
            # HINT: Can any function from the Scene class help?
            # HINT: Set bg_colour to (0.0, 0.0, 0.0)
            # HINT: Set img_size to (128, 128)
            # HINT: Get per_splat from args.gaussians_per_splat
            # HINT: camera is available above
            pred_img, depth, mask = scene.render(camera=camera, per_splat=args.gaussians_per_splat, img_size=(gt_img.shape[0], gt_img.shape[1]))

            gt_npy = gt_img.detach().cpu().numpy()
            pred_npy = pred_img.detach().cpu().numpy()
            psnr = peak_signal_noise_ratio(gt_npy, pred_npy)
            ssim_metric = structural_similarity(gt_npy, pred_npy, channel_axis=-1, data_range=1.0)

            psnr_vals.append(psnr)
            ssim_vals.append(ssim_metric)

    mean_psnr = np.mean(psnr_vals)
    mean_ssim = np.mean(ssim_vals)
    print(f"[*] Evaluation --- Mean PSNR: {mean_psnr:.3f}")
    print(f"[*] Evaluation --- Mean SSIM: {mean_ssim:.3f}")

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_path", default="./output", type=str,
        help="Path to the directory where output should be saved to."
    )
    parser.add_argument(
        "--data_path", default="./data/materials", type=str,
        help="Path to the dataset."
    )
    parser.add_argument(
        "--gaussians_per_splat", default=-1, type=int,
        help=(
            "Number of gaussians to splat in one function call. If set to -1, "
            "then all gaussians in the scene are splat in a single function call. "
            "If set to any other positive interger, then it determines the number of "
            "gaussians to splat per function call (the last function call might splat "
            "lesser number of gaussians). In general, the algorithm can run faster "
            "if more gaussians are splat per function call, but at the cost of higher GPU "
            "memory consumption."
        )
    )
    parser.add_argument(
        "--num_itrs", default=1000, type=int,
        help="Number of iterations to train the model."
    )
    parser.add_argument(
        "--viz_freq", default=20, type=int,
        help="Frequency with which visualization should be performed."
    )
    parser.add_argument("--device", default="cuda", type=str, choices=["cuda", "cpu"])
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()
    run_training(args)
