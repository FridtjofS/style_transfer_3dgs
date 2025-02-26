import torch
import torch.nn.functional as F
from gsplat import quat_scale_to_covar_preci


def gaussian_decomp(gaussians, viewpoint_camera, input_mask, indices_mask):
    xyz = gaussians.means
    point_image = project_to_2d(viewpoint_camera, xyz)

    conv2d = conv2d_matrix(gaussians, viewpoint_camera, indices_mask, device="cuda")
    height = viewpoint_camera.height.item()
    width = viewpoint_camera.width.item()
    index_in_all, ratios, dir_vector = compute_ratios(
        conv2d, point_image, indices_mask, input_mask, height, width
    )

    # decomp_gaussians = update(
    #     gaussians, viewpoint_camera, index_in_all, ratios, dir_vector
    # )

    selected_index, new_xyz, new_scaling = update(
        gaussians, viewpoint_camera, index_in_all, ratios, dir_vector
    )
    return selected_index, new_xyz, new_scaling


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


## assume obtain 2d convariance matrx: N, 2, 2
def compute_ratios(conv_2d, points_xy, indices_mask, sam_mask, h, w):
    means = points_xy[indices_mask]
    # 计算特征值和特征向量
    eigvals, eigvecs = torch.linalg.eigh(conv_2d)
    # 判断长轴
    max_eigval, max_idx = torch.max(eigvals, dim=1)
    max_eigvec = torch.gather(
        eigvecs, dim=1, index=max_idx.unsqueeze(1).unsqueeze(2).repeat(1, 1, 2)
    )  # (N, 1, 2)最大特征向量
    # 3 sigma，计算两个顶点的坐标
    long_axis = torch.sqrt(max_eigval) * 3
    max_eigvec = max_eigvec.squeeze(1)
    max_eigvec = max_eigvec / torch.norm(max_eigvec, dim=1).unsqueeze(-1)
    vertex1 = means + 0.5 * long_axis.unsqueeze(1) * max_eigvec
    vertex2 = means - 0.5 * long_axis.unsqueeze(1) * max_eigvec
    vertex1 = torch.clip(
        vertex1,
        torch.tensor([0, 0]).to(points_xy.device),
        torch.tensor([w - 1, h - 1]).to(points_xy.device),
    )
    vertex2 = torch.clip(
        vertex2,
        torch.tensor([0, 0]).to(points_xy.device),
        torch.tensor([w - 1, h - 1]).to(points_xy.device),
    )
    # 得到每个gaussian顶点的label
    vertex1_xy = torch.round(vertex1).long()
    vertex2_xy = torch.round(vertex2).long()
    vertex1_label = sam_mask[vertex1_xy[:, 1].to("cpu"), vertex1_xy[:, 0].to("cpu")]
    vertex2_label = sam_mask[vertex2_xy[:, 1].to("cpu"), vertex2_xy[:, 0].to("cpu")]
    # 得到需要调整gaussian的索引  还有一种情况 中心在mask内，但是两个端点在mask以外
    index = torch.nonzero(vertex1_label ^ vertex2_label, as_tuple=True)[0]
    # special_index = (vertex1_label == 0) & (vertex2_label == 0)
    # index = torch.cat((index, special_index), dim=0)
    selected_vertex1_xy = vertex1_xy[index]
    selected_vertex2_xy = vertex2_xy[index]
    # 找到2D 需要平移的方向, 用一个符号函数，1表示沿着特征向量方向，-1表示相反
    sign_direction = vertex1_label[index] - vertex2_label[index]
    direction_vector = max_eigvec[index].to(
        sign_direction.device
    ) * sign_direction.unsqueeze(-1).to(sign_direction.device)

    # 两个顶点连线上的像素点
    ratios = []
    for k in range(len(index)):
        x1, y1 = selected_vertex1_xy[k]
        x2, y2 = selected_vertex2_xy[k]
        # print(k, x1, x2)
        if x1 < x2:
            x_point = torch.arange(x1, x2 + 1).to(points_xy.device)
            y_point = y1 + (y2 - y1) / (x2 - x1) * (x_point - x1)
        elif x1 < x2:
            x_point = torch.arange(x2, x1 + 1).to(points_xy.device)
            y_point = y1 + (y2 - y1) / (x2 - x1) * (x_point - x1)
        else:
            if y1 < y2:
                y_point = torch.arange(y1, y2 + 1).to(points_xy.device)
                x_point = torch.ones_like(y_point) * x1
            else:
                y_point = torch.arange(y2, y1 + 1).to(points_xy.device)
                x_point = torch.ones_like(y_point) * x1

        x_point = torch.round(torch.clip(x_point, 0, w - 1)).long()
        y_point = torch.round(torch.clip(y_point, 0, h - 1)).long()
        # print(x_point.max(), y_point.max())
        # 判断连线上的像素是否在sam mask之内, 计算所占比例
        in_mask = sam_mask[y_point.to("cpu"), x_point.to("cpu")]
        ratios.append(sum(in_mask) / len(in_mask))

    ratios = torch.tensor(ratios)
    # 在3D Gaussian中对这些gaussians做调整，xyz和scaling
    index_in_all = indices_mask[index]

    return index_in_all, ratios, direction_vector


def conv2d_matrix(gaussians, viewpoint_camera, indices_mask, device):
    # 3d convariance matrix
    quats = gaussians.quats[indices_mask]
    scales = gaussians.scales[indices_mask]
    conv3d_matrix, _ = quat_scale_to_covar_preci(
        quats, torch.exp(scales), compute_preci=False
    )
    # conv3d = gaussians.get_covariance(scaling_modifier=1)[indices_mask]
    # conv3d_matrix = compute_conv3d(conv3d).to(device)

    w2c = get_viewmat(viewpoint_camera.camera_to_worlds)[0].cuda().transpose(0, 1)
    mask_xyz = gaussians.means[indices_mask]
    pad_mask_xyz = F.pad(input=mask_xyz, pad=(0, 1), mode="constant", value=1)
    t = pad_mask_xyz @ w2c[:, :3]  # N, 3
    height = viewpoint_camera.height.item()
    width = viewpoint_camera.width.item()
    # tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    # tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    focal_x = viewpoint_camera.fx.to(device)
    focal_y = viewpoint_camera.fy.to(device)

    tanfovx = height / (2 * focal_x)
    tanfovy = width / (2 * focal_y)

    lim_xy = torch.tensor([1.3 * tanfovx, 1.3 * tanfovy]).to(device)
    t[:, :2] = (
        torch.clip(t[:, :2] / t[:, 2, None], -1.0 * lim_xy, lim_xy) * t[:, 2, None]
    )
    J_matrix = torch.zeros((mask_xyz.shape[0], 3, 3)).to(device)
    J_matrix[:, 0, 0] = focal_x / t[:, 2]
    J_matrix[:, 0, 2] = -1 * (focal_x * t[:, 0]) / (t[:, 2] * t[:, 2])
    J_matrix[:, 1, 1] = focal_y / t[:, 2]
    J_matrix[:, 1, 2] = -1 * (focal_y * t[:, 1]) / (t[:, 2] * t[:, 2])
    W_matrix = w2c[:3, :3]  # 3,3
    T_matrix = (W_matrix @ J_matrix.permute(1, 2, 0)).permute(2, 0, 1)  # N,3,3

    conv2d_matrix = torch.bmm(
        T_matrix.permute(0, 2, 1), torch.bmm(conv3d_matrix, T_matrix)
    )[:, :2, :2]

    return conv2d_matrix


def update(gaussians, view, selected_index, ratios, dir_vector):
    ratios = ratios.unsqueeze(-1).to("cuda")
    selected_xyz = gaussians.means[selected_index]
    selected_scaling = torch.exp(gaussians.scales[selected_index])
    selected_quats = gaussians.quats[selected_index]
    conv3d, _ = quat_scale_to_covar_preci(
        selected_quats, selected_scaling, compute_preci=False
    )
    conv3d_matrix = conv3d.to("cuda")
    # conv3d = gaussians.get_covariance(scaling_modifier=1)[selected_index]
    # conv3d_matrix = compute_conv3d(conv3d).to("cuda")

    # 计算特征值和特征向量
    eigvals, eigvecs = torch.linalg.eigh(conv3d_matrix)
    # 判断长轴
    max_eigval, max_idx = torch.max(eigvals, dim=1)
    max_eigvec = torch.gather(
        eigvecs, dim=1, index=max_idx.unsqueeze(1).unsqueeze(2).repeat(1, 1, 3)
    )  # (N, 1, 3)最大特征向量
    long_axis = torch.sqrt(max_eigval) * 3
    max_eigvec = max_eigvec.squeeze(1)
    max_eigvec = max_eigvec / torch.norm(max_eigvec, dim=1).unsqueeze(-1)
    new_scaling = selected_scaling * ratios * 0.8
    # new_scaling = selected_scaling

    # 更新原gaussians里面相应的点，有两个方向，需要判断哪个方向:
    # 把3d特征向量投影到2d，与2d的平移方向计算内积，大于0表示正方向，小于0表示负方向
    max_eigvec_2d = project_to_2d(view, max_eigvec)
    sign_direction = torch.sum(
        max_eigvec_2d.to(dir_vector.device) * dir_vector, dim=1
    ).unsqueeze(-1)
    sign_direction = torch.where(sign_direction > 0, 1, -1)
    new_xyz = (
        selected_xyz.to(sign_direction.device)
        + 0.5
        * (1 - ratios.to(sign_direction.device))
        * long_axis.unsqueeze(1).to(sign_direction.device)
        * max_eigvec.to(sign_direction.device)
        * sign_direction
    )

    # gaussians.means = gaussians.means.detach().clone().requires_grad_(False)
    # gaussians.scales = gaussians.scales.detach().clone().requires_grad_(False)
    # gaussians.means[selected_index] = new_xyz.to(gaussians.means.device)
    # gaussians.scales[selected_index] = torch.log(new_scaling).to(
    #    gaussians.scales.device
    # )

    return selected_index, new_xyz, new_scaling
