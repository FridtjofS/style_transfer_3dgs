import torch
import numpy as np

import torch.nn.functional as F
import torchvision.transforms.functional as func
import torchvision

# from SAGS.seg_utils import conv2d_matrix, compute_ratios, update
from nerfstudio.cameras.cameras import Cameras


def get_viewmat(optimized_camera_to_world):
    """
    function that converts c2w to gsplat world2camera matrix, using compile for some speed
    """
    R = optimized_camera_to_world[:, :3, :3]  # 3 x 3
    T = optimized_camera_to_world[:, :3, 3:4]  # 3 x 1
    # flip the z and y axes to align with gsplat conventions
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0  # homogenous
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat


def get_3d_prompts(prompts_2d, point_image, xyz, depth=None):
    r = 4
    x_range = torch.arange(prompts_2d[0] - r, prompts_2d[0] + r)
    y_range = torch.arange(prompts_2d[1] - r, prompts_2d[1] + r)
    x_grid, y_grid = torch.meshgrid(x_range, y_range)
    neighbors = torch.stack([x_grid, y_grid], dim=2).reshape(-1, 2).to("cuda")
    prompts_index = [torch.where((point_image == p).all(dim=1))[0] for p in neighbors]
    indexs = []
    for index in prompts_index:
        if index.nelement() > 0:
            indexs.append(index)
    indexs = torch.unique(torch.cat(indexs, dim=0))
    indexs_depth = depth[indexs]
    valid_depth = indexs_depth[indexs_depth > 0]
    _, sorted_indices = torch.sort(valid_depth)
    valid_indexs = indexs[depth[indexs] > 0][sorted_indices[0]]

    return xyz[valid_indexs][:3].unsqueeze(0)


## Given 1st view point prompts, find corresponding 3D Gaussian point prompts
def generate_3d_prompts(xyz, viewpoint_camera: Cameras, prompts_2d):
    w2c_matrix = get_viewmat(viewpoint_camera.camera_to_worlds)[0].cuda()  # 4x4
    intrinsics = torch.eye(4, device="cuda")
    intrinsics[:3, :3] = viewpoint_camera.get_intrinsics_matrices().squeeze()
    # intrinsics = viewpoint_camera.get_intrinsics_matrices()
    full_matrix = (intrinsics @ w2c_matrix).transpose(0, 1)  # 4x4
    # project to image plane
    xyz = F.pad(input=xyz, pad=(0, 1), mode="constant", value=1)
    p_hom = (xyz @ full_matrix).transpose(0, 1)  # N, 4 -> 4, N
    p_w = 1.0 / (p_hom[-2, :] + 0.0000001)
    p_proj = p_hom[:2, :] * p_w

    p_view = (xyz @ w2c_matrix.transpose(0, 1)[:, :3]).transpose(0, 1)  # N, 3 -> 3, N
    depth = p_view[-1, :].detach().clone()
    # valid_depth = depth >= 0
    point_image = p_proj.detach().clone()
    point_image = torch.round(point_image.transpose(0, 1)).long()

    prompts_2d = torch.tensor(prompts_2d).to("cuda")
    prompts_3d = []
    for i in range(prompts_2d.shape[0]):
        prompts_3d.append(get_3d_prompts(prompts_2d[i], point_image, xyz, depth))
    prompts_3D = torch.cat(prompts_3d, dim=0)

    return prompts_3D


## Project 3D points to 2D plane
def project_to_2d(viewpoint_camera, points3D):
    w2c_matrix = get_viewmat(viewpoint_camera.camera_to_worlds)[0].cuda()  # 4x4
    intrinsics = torch.eye(4, device="cuda")
    intrinsics[:3, :3] = viewpoint_camera.get_intrinsics_matrices()
    # intrinsics = viewpoint_camera.get_intrinsics_matrices()
    full_matrix = (intrinsics @ w2c_matrix).transpose(0, 1)  # 4x4
    # project to image plane
    if points3D.shape[-1] != 4:
        points3D = F.pad(input=points3D, pad=(0, 1), mode="constant", value=1)
    p_hom = (points3D @ full_matrix).transpose(0, 1)  # N, 4 -> 4, N
    p_w = 1.0 / (p_hom[-2, :] + 0.0000001)
    p_proj = p_hom[:2, :] * p_w

    # p_view = (xyz @ w2c_matrix.transpose(0, 1)[:, :3]).transpose(0, 1)  # N, 3 -> 3, N
    # depth = p_view[-1, :].detach().clone()
    # valid_depth = depth >= 0

    point_image = p_proj.detach().clone()
    point_image = torch.round(point_image.transpose(0, 1))

    return point_image


## Single view assignment
def mask_inverse(xyz, viewpoint_camera, sam_mask):
    w2c_matrix = get_viewmat(viewpoint_camera.camera_to_worlds)[0].cuda()  # 4x4
    # project to camera space
    xyz = F.pad(input=xyz, pad=(0, 1), mode="constant", value=1)
    p_view = (xyz @ w2c_matrix.transpose(0, 1)[:, :3]).transpose(0, 1)  # N, 3 -> 3, N
    depth = p_view[-1, :].detach().clone()
    valid_depth = depth >= 0

    h = viewpoint_camera.image_height.squeeze().item()
    w = viewpoint_camera.image_width.squeeze().item()

    if sam_mask.shape[0] != h or sam_mask.shape[1] != w:
        sam_mask = (
            func.resize(sam_mask.unsqueeze(0), (h, w), antialias=True).squeeze(0).long()
        )
    else:
        sam_mask = sam_mask.long()

    point_image = project_to_2d(viewpoint_camera, xyz)
    point_image = point_image.long()

    # 判断x,y是否在图像范围之内
    valid_x = (point_image[:, 0] >= 0) & (point_image[:, 0] < w)
    valid_y = (point_image[:, 1] >= 0) & (point_image[:, 1] < h)
    valid_mask = valid_x & valid_y & valid_depth
    point_mask = torch.full((point_image.shape[0],), -1)

    point_mask[valid_mask.cpu()] = sam_mask[
        point_image[valid_mask.cpu(), 1].cpu(), point_image.cpu()[valid_mask.cpu(), 0]
    ]
    indices_mask = torch.where(point_mask == 1)[0]

    return point_mask, indices_mask


## Multi-view label voting
def ensemble(multiview_masks, threshold=0.7):
    # threshold = 0.7
    multiview_masks = torch.cat(multiview_masks, dim=1)
    vote_labels, _ = torch.mode(multiview_masks, dim=1)
    # # select points with score > threshold
    matches = torch.eq(multiview_masks, vote_labels.unsqueeze(1))
    ratios = torch.sum(matches, dim=1) / multiview_masks.shape[1]
    ratios_mask = ratios > threshold
    labels_mask = (vote_labels == 1) & ratios_mask
    indices_mask = torch.where(labels_mask)[0].detach().cpu()

    return vote_labels, indices_mask


def first_axis_intersection(camera_position, look_direction):
    """
    Compute the intersection of the look direction with the first axis (x, y, or z) in the positive
    direction starting from the camera position.

    Args:
        camera_position (tuple): The camera position.
        look_direction (tuple): The look direction.

    Returns:
        tuple: The intersection point.

    """
    camera_position = np.array(camera_position)
    look_direction = np.array(look_direction)
    
    # Avoid division by zero
    epsilon = 1e-6
    
    # Compute t for each plane
    t_x = -camera_position[0] / (look_direction[0] + epsilon) if look_direction[0] != 0 else np.inf
    t_y = -camera_position[1] / (look_direction[1] + epsilon) if look_direction[1] != 0 else np.inf
    t_z = -camera_position[2] / (look_direction[2] + epsilon) if look_direction[2] != 0 else np.inf
    
    # Collect valid t values
    t_values = [t for t in [t_x, t_y, t_z] if t > 0]
    
    if not t_values:
        # No axis intersection (look direction is away from the center)
        return camera_position + look_direction
    
    # get the one closest to [0, 0, 0]
    min_intersections = [camera_position + t * look_direction for t in t_values]

    min_idx = np.argmin([np.mean(np.abs(intersection)) for intersection in min_intersections])

    return min_intersections[min_idx]
    
    # Get the first intersection
    t_min = min(t_values)


    intersection_point = camera_position + t_min * look_direction
    
    return intersection_point

# Define the L1 loss
def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

# Define the L2 loss
def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def argmin_cos_distance(a, b, center=False):
    """
    a: [b, c, hw],
    b: [b, c, h2w2]
    """
    if center:
        a = a - a.mean(2, keepdims=True)
        b = b - b.mean(2, keepdims=True)

    b_norm = ((b * b).sum(1, keepdims=True) + 1e-8).sqrt()
    b = b / (b_norm + 1e-8)

    z_best = []
    loop_batch_size = int(1e8 / b.shape[-1])
    for i in range(0, a.shape[-1], loop_batch_size):
        a_batch = a[..., i : i + loop_batch_size]
        a_batch_norm = ((a_batch * a_batch).sum(1, keepdims=True) + 1e-8).sqrt()
        a_batch = a_batch / (a_batch_norm + 1e-8)

        d_mat = 1.0 - torch.matmul(a_batch.transpose(2, 1), b)

        z_best_batch = torch.argmin(d_mat, 2)
        z_best.append(z_best_batch)
    z_best = torch.cat(z_best, dim=-1)

    return z_best


def nn_feat_replace(a, b):
    n, c, h, w = a.size()
    n2, c, h2, w2 = b.size()

    assert (n == 1) and (n2 == 1)

    a_flat = a.view(n, c, -1)
    b_flat = b.view(n2, c, -1)
    b_ref = b_flat.clone()

    z_new = []
    for i in range(n):
        z_best = argmin_cos_distance(a_flat[i : i + 1], b_flat[i : i + 1])
        z_best = z_best.unsqueeze(1).repeat(1, c, 1)
        feat = torch.gather(b_ref, 2, z_best)
        z_new.append(feat)

    z_new = torch.cat(z_new, 0)
    z_new = z_new.view(n, c, h, w)
    return z_new


def cos_loss(a, b):
    a_norm = (a * a).sum(1, keepdims=True).sqrt()
    b_norm = (b * b).sum(1, keepdims=True).sqrt()
    a_tmp = a / (a_norm + 1e-8)
    b_tmp = b / (b_norm + 1e-8)
    cossim = (a_tmp * b_tmp).sum(1)
    cos_d = 1.0 - cossim
    return cos_d.mean()

def content_loss(x_feats, s_feats):
    """
    Computes the content loss (MSE between the feature maps).
    x_feats: feature maps from the generated image
    s_feats: feature maps from the style image (target content features)
    """
    return F.mse_loss(x_feats, s_feats)

def gram_matrix(x):
    """
    Computes the Gram matrix of a given feature map.
    x: feature map (B, C, H, W) where B is batch size, C is number of channels,
       H is height, W is width
    """
    b, c, h, w = x.size()

    features = x.view(b, c, h * w)  # Reshape the feature map to (B*C, H*W)
    gram = torch.bmm(features, features.transpose(1, 2))  # Compute the Gram matrix
    gram /= (h * w)  # Normalize the Gram matrix
    return gram

def gram_loss(x_feats, s_feats):
    """
    Computes the style loss (MSE between the Gram matrices).
    x_feats: feature maps from the generated image
    s_feats: feature maps from the style image
    """
    G_x = gram_matrix(x_feats)
    G_s = gram_matrix(s_feats)

    # weight the loss by the number of elements in the feature map
    return F.mse_loss(G_x, G_s)



class NNFMLoss(torch.nn.Module):
    def __init__(self, device):
        super().__init__()

        self.vgg = torchvision.models.vgg16(pretrained=True).eval().to(device)
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def get_feats(self, x, layers=[]):
        x = self.normalize(x)
        final_ix = max(layers)
        outputs = []

        for ix, layer in enumerate(self.vgg.features):
            x = layer(x)
            if ix in layers:
                outputs.append(x)

            if ix == final_ix:
                break

        return outputs

    def forward(
        self,
        outputs,
        styles,
        blocks=[2],
        loss_names=["nnfm_loss"],  # can also include 'gram_loss', 'content_loss', 'l1_loss', 'l2_loss'
        contents=None,
    ):
        for x in loss_names:
            assert x in ["nnfm_loss", "content_loss", "gram_loss", "l1_loss", "l2_loss"]

        # Define block indexes for each type of loss
        block_indexes = {
            "content_loss": [[1, 3], [6, 8], [11, 13, 15]],  # Shallow to intermediate layers
            "gram_loss": [[6, 8], [11, 13, 15], [18, 20, 22]],  # Intermediate layers for style
            "nnfm_loss": [[1, 3], [6, 8], [11, 13, 15], [18, 20, 22], [25, 27, 29]],  # All layers for nnfm_loss
            "l1_loss": [[1, 3], [6, 8], [11, 13, 15], [18, 20, 22], [25, 27, 29]],  # Can apply to all layers
            "l2_loss": [[1, 3], [6, 8], [11, 13, 15], [18, 20, 22], [25, 27, 29]],  # Can apply to all layers
        }

        weighting_dict = {
            "content_loss": [1.0, 0.8, 0.5],  # Higher weight for shallower layers
            "gram_loss": [0.14, 0.28, 0.58],  # Higher weight for deeper style layers
            "nnfm_loss": [0.33, 0.27, 0.2, 0.13, 0.07],  # Gradually decreasing across all layers
            "l1_loss": [1.0, 0.9, 0.7, 0.5, 0.3],  # Similar decay as nnfm_loss
            "l2_loss": [1.0, 0.9, 0.7, 0.5, 0.3],  # Same as l1_loss
        }


        # Sort the blocks and combine the layers for the selected blocks
        blocks.sort()
        all_layers = []
        for loss_name in loss_names:
            for block in blocks:
                all_layers += block_indexes[loss_name][block]

        x_feats_all = self.get_feats(outputs, all_layers)
        with torch.no_grad():
            s_feats_all = self.get_feats(styles, all_layers)

        ix_map = {}
        for a, b in enumerate(all_layers):
            ix_map[b] = a

        # Initialize loss dictionary
        loss_dict = dict([(x, 0.0) for x in loss_names])

        # Loop through blocks to compute losses
        for block in blocks:
            for loss_name in loss_names:
                layers = block_indexes[loss_name][block]
                x_feats = torch.cat([x_feats_all[ix_map[ix]] for ix in layers], 1)
                s_feats = torch.cat([s_feats_all[ix_map[ix]] for ix in layers], 1)

                if loss_name == "nnfm_loss":
                    target_feats = nn_feat_replace(x_feats, s_feats)
                    loss_dict["nnfm_loss"] += cos_loss(x_feats, target_feats) * weighting_dict["nnfm_loss"][block]
                if loss_name == "l1_loss":
                    target_feats = nn_feat_replace(x_feats, s_feats)
                    loss_dict["l1_loss"] += l1_loss(x_feats, target_feats) * weighting_dict["l1_loss"][block]
                if loss_name == "l2_loss":
                    target_feats = nn_feat_replace(x_feats, s_feats)
                    loss_dict["l2_loss"] += l2_loss(x_feats, target_feats) * weighting_dict["l2_loss"][block]
                if loss_name == "content_loss":
                    loss_dict["content_loss"] += content_loss(x_feats, s_feats) * weighting_dict["content_loss"][block]
                if loss_name == "gram_loss":
                    loss_dict["gram_loss"] += gram_loss(x_feats, s_feats) * weighting_dict["gram_loss"][block]

        return loss_dict
    

    
