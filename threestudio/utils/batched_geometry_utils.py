# Code from SPARF: https://github.com/google-research/sparf

from typing import Any, List, Union, Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def pose_inverse_4x4(mat: torch.Tensor, use_inverse: bool = False) -> torch.Tensor:
    """
    Transforms world2cam into cam2world or vice-versa, without computing the inverse.
    Args:
        mat (torch.Tensor): pose matrix (B, 4, 4) or (4, 4)
    """
    # invert a camera pose
    out_mat = torch.zeros_like(mat)

    if len(out_mat.shape) == 3:
        # must be (B, 4, 4)
        out_mat[:, 3, 3] = 1
        R, t = mat[:, :3, :3], mat[:, :3, 3:]
        R_inv = R.inverse() if use_inverse else R.transpose(-1, -2)
        t_inv = (-R_inv @ t)[..., 0]

        pose_inv = torch.cat([R_inv, t_inv[..., None]], dim=-1)  # [...,3,4]

        out_mat[:, :3] = pose_inv
    else:
        out_mat[3, 3] = 1
        R, t = mat[:3, :3], mat[:3, 3:]
        R_inv = R.inverse() if use_inverse else R.transpose(-1, -2)
        t_inv = (-R_inv @ t)[..., 0]
        pose_inv = torch.cat([R_inv, t_inv[..., None]], dim=-1)  # [3,4]
        out_mat[:3] = pose_inv
    # assert torch.equal(out_mat, torch.inverse(mat))
    return out_mat


def to_homogeneous(points: Union[torch.Tensor, np.ndarray]):
    """Convert N-dimensional points to homogeneous coordinates.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N).
    Returns:
        A torch.Tensor or numpy.ndarray with size (..., N+1).
    """
    if isinstance(points, torch.Tensor):
        pad = points.new_ones(points.shape[:-1] + (1,))
        return torch.cat([points, pad], dim=-1)
    elif isinstance(points, np.ndarray):
        pad = np.ones((points.shape[:-1] + (1,)), dtype=points.dtype)
        return np.concatenate([points, pad], axis=-1)
    else:
        raise ValueError


def from_homogeneous(points: Union[torch.Tensor, np.ndarray]):
    """Remove the homogeneous dimension of N-dimensional points.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N+1).
    Returns:
        A torch.Tensor or numpy ndarray with size (..., N).
    """
    return points[..., :-1] / (points[..., -1:] + 1e-6)


def batched_eye_like(x: torch.Tensor, n):
    """Create a batch of identity matrices.
    Args:
        x: a reference torch.Tensor whose batch dimension will be copied.
        n: the size of each identity matrix.
    Returns:
        A torch.Tensor of size (B, n, n), with same dtype and device as x.
    """
    return torch.eye(n).to(x)[None].repeat(len(x), 1, 1)


def create_norm_matrix(shift, scale):
    """Create a normalization matrix that shifts and scales points."""
    T = batched_eye_like(shift, 3)
    T[:, 0, 0] = T[:, 1, 1] = scale
    T[:, :2, 2] = shift
    return T


def normalize_keypoints_with_intrinsics(kpts: torch.Tensor, K0: torch.Tensor):
    """
    Normalizes a set of 2D keypoints
    Args:
        kpts: a batch of N 2-dimensional keypoints: (B, N, 2).
        K0: a batch of 3x3 intrinsic matrices: (B, 3, 3)
    """
    kpts[..., 0] -= K0[..., 0, 2].unsqueeze(1)
    kpts[..., 1] -= K0[..., 1, 2].unsqueeze(1)

    kpts[..., 0] /= K0[..., 0, 0].unsqueeze(1)
    kpts[..., 1] /= K0[..., 1, 1].unsqueeze(1)
    return kpts


def normalize_keypoints(kpts: torch.Tensor, size=None, shape=None):
    """Normalize a set of 2D keypoints for input to a neural network.
    Perform the normalization according to the size of the corresponding
    image: shift by half and scales by the longest edge.
    Use either the image size or its tensor shape.
    Args:
        kpts: a batch of N D-dimensional keypoints: (B, N, D).
        size: a tensor of the size the image `[W, H]`.
        shape: a tuple of the image tensor shape `(B, C, H, W)`.
    """
    if size is None:
        assert shape is not None
        _, _, h, w = shape
        one = kpts.new_tensor(1)
        size = torch.stack([one * w, one * h])[None]

    shift = size.float() / 2
    scale = size.max(1).values.float() / 2  # actual SuperGlue mult by 0.7
    kpts = (kpts - shift[:, None]) / scale[:, None, None]

    T_norm = create_norm_matrix(shift, scale)
    T_norm_inv = create_norm_matrix(-shift / scale[:, None], 1.0 / scale)
    return kpts, T_norm, T_norm_inv


def batched_scale_intrinsics(
    K: torch.Tensor, scales: torch.Tensor, invert_scales=True
) -> torch.Tensor:
    """
    Args:
        K: a batch of N 3x3 intrinsic matrix, (N, 3, 3)
        scales: a tensor of the shape (N, 1, 1), first horizontal
    """
    scaling_mat = batched_eye_like(K, 3)
    if invert_scales:
        scaling_mat[:, 0, 0] = 1.0 / scales[0]
        scaling_mat[:, 1, 1] = 1.0 / scales[1]
    else:
        scaling_mat[:, 0, 0] = scales[0]
        scaling_mat[:, 1, 1] = scales[1]
    return scaling_mat @ K  # scales @ K


def sample_depth(
    pts: torch.Tensor, depth: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """sample depth at points.

    Args:
        pts (torch.Tensor): (N, 2)
        depth (torch.Tensor): (B, 1, H, W)
    """
    h, w = depth.shape[-2:]
    grid_sample = torch.nn.functional.grid_sample
    batched = len(depth.shape) == 3
    if not batched:
        pts, depth = pts[None], depth[None]

    pts = (pts / pts.new_tensor([[w - 1, h - 1]]) * 2 - 1)[:, None]
    depth = torch.where(depth > 0, depth, depth.new_tensor(float("nan")))
    depth = depth[:, None]
    interp_lin = grid_sample(depth, pts, align_corners=True, mode="bilinear")[:, 0, 0]
    interp_nn = grid_sample(depth, pts, align_corners=True, mode="nearest")[:, 0, 0]
    interp = torch.where(torch.isnan(interp_lin), interp_nn, interp_lin)
    valid = (~torch.isnan(interp)) & (interp > 0)
    # will exclude out of view matches here, except if the depth was dense and the points falls right at the border,
    # then nearest can get the depth value.
    if not batched:
        interp, valid = interp[0], valid[0]
    return interp, valid


def batch_project_to_other_img_and_check_depth(
    kpi: torch.Tensor,
    di: torch.Tensor,
    depthj: torch.Tensor,
    Ki: torch.Tensor,
    Kj: torch.Tensor,
    T_itoj: torch.Tensor,
    validi: torch.Tensor,
    rth: float = 0.1,
    return_repro_error: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project pixels of one image to the other, and run depth-check.
    Args:
        kpi: BxNx2 coordinates in pixels of image i
        di: BxN, corresponding depths of image i
        depthj: depth map of image j, BxHxW
        Ki: intrinsics of image i, Bx3x3
        Kj: intrinsics of image j, Bx3x3
        T_itoj: Transform matrix from coordinate system of i to j, Bx4x4
        validi: BxN, Bool mask
        rth: percentage of acceptable depth reprojection error.
        return_repro_error: Bool

    Returns:
        kpi_j: Pixels projection in image j, BxNx2
        visible: Bool mask, visible pixels that have a valid reprojection error, BxN
    """

    kpi_3d_i = to_homogeneous(kpi) @ torch.inverse(Ki).transpose(-1, -2)
    kpi_3d_i = kpi_3d_i * di[..., None]  # non-homogeneous coordinates
    kpi_3d_j = from_homogeneous(to_homogeneous(kpi_3d_i) @ T_itoj.transpose(-1, -2))
    kpi_j = from_homogeneous(kpi_3d_j @ Kj.transpose(-1, -2))
    di_j = kpi_3d_j[..., -1]

    dj, validj = sample_depth(kpi_j, depthj)
    repro_error = torch.abs(di_j - dj) / dj
    consistent = repro_error < rth
    visible = validi & consistent & validj
    if return_repro_error:
        return kpi_j, visible, repro_error
    return kpi_j, visible


def batch_project_to_other_img(
    kpi: torch.Tensor,
    di: torch.Tensor,
    Ki: torch.Tensor,
    Kj: torch.Tensor,
    src_c2w: torch.Tensor,
    trg_c2w: torch.Tensor,
    return_depth=False,
) -> torch.Tensor:
    """
    Project pixels of one image to the other.
    Args:
        kpi: BxNx2 coordinates in pixels of image i
        di: BxN, corresponding depths of image i
        Ki: intrinsics of image i, Bx3x3
        Kj: intrinsics of image j, Bx3x3
        T_itoj: Transform matrix from coordinate system of i to j, Bx4x4
        return_depth: Bool

    Returns:
        kpi_j: Pixels projection in image j, BxNx2
        di_j: Depth of the projections in image j, BxN
    """
    if len(di.shape) == len(kpi.shape):
        # di must be BxNx1
        di = di.squeeze(-1)
    kpi_3d_i = to_homogeneous(kpi) @ torch.inverse(Ki).transpose(-1, -2)
    kpi_3d_i_d = kpi_3d_i * di[..., None]  # non-homogeneous coordinates
    kpi_3d_i_h = to_homogeneous(kpi_3d_i_d)
    kpi_3d_j_h = kpi_3d_i_h @ (pose_inverse_4x4(trg_c2w) @ src_c2w).transpose(-1, -2)
    kpi_3d_j = from_homogeneous(kpi_3d_j_h)
    kpi_j = from_homogeneous(kpi_3d_j @ Kj.transpose(-1, -2))
    if return_depth:
        di_j = kpi_3d_j[..., -1]
        return kpi_j, di_j
    return kpi_j


def batch_backproject_to_3d(
    kpi: torch.Tensor, di: torch.Tensor, Ki: torch.Tensor, T_itoj: torch.Tensor
) -> torch.Tensor:
    """
    Backprojects pixels to 3D space
    Args:
        kpi: BxNx2 coordinates in pixels
        di: BxN, corresponding depths
        Ki: camera intrinsics, Bx3x3
        T_itoj: Bx4x4
    Returns:
        kpi_3d_j: 3D points in coordinate system j, BxNx3
    """

    kpi_3d_i = to_homogeneous(kpi) @ torch.inverse(Ki).transpose(-1, -2)
    kpi_3d_i = kpi_3d_i * di[..., None]  # non-homogeneous coordinates
    kpi_3d_j = from_homogeneous(to_homogeneous(kpi_3d_i) @ T_itoj.transpose(-1, -2))
    return kpi_3d_j  # Nx3


def batch_project(
    kpi_3d_i: torch.Tensor, T_itoj: torch.Tensor, Kj: torch.Tensor, return_depth=False
):
    """
    Projects 3D points to image pixels coordinates.
    Args:
        kpi_3d_i: 3D points in coordinate system i, BxNx3
        T_itoj: Bx4x4
        Kj: camera intrinsics Bx3x3

    Returns:
        pixels projections in image j, BxNx2
    """
    kpi_3d_in_j = from_homogeneous(to_homogeneous(kpi_3d_i) @ T_itoj.transpose(-1, -2))
    kpi_2d_in_j = kpi_3d_in_j @ Kj.transpose(-1, -2)
    if return_depth:
        return from_homogeneous(kpi_2d_in_j), kpi_3d_in_j[..., -1]
    return from_homogeneous(kpi_2d_in_j)


def batch_check_depth_reprojection_error(
    kpi_3d_i: torch.Tensor,
    depthj: torch.Tensor,
    Kj: torch.Tensor,
    T_itoj: torch.Tensor,
    validi: torch.Tensor = None,
    rth=0.1,
) -> torch.Tensor:
    """
    Run depth consistency check. Project the 3D point to image j and compare depth of the projection
    to available depth measured at image j.
    Args:
        kpi_3d_i: 3D points in coordinate system i, BxNx3
        depthj: depth map of image j, BxHxW
        Kj: Bx3x3
        T_itoj: Bx4x4
        validi: valid 3d poitn mask, BxN, bool
        rth: percentage of depth error acceptable

    Returns:
        visible: torch.Bool BxN, indicates if the 3D point has a valib depth reprojection error.
    """
    kpi_3d_j = from_homogeneous(to_homogeneous(kpi_3d_i) @ T_itoj.transpose(-1, -2))
    kpi_j = from_homogeneous(kpi_3d_j @ Kj.transpose(-1, -2))
    di_j = kpi_3d_j[..., -1]

    dj, validj = sample_depth(kpi_j, depthj)

    consistent = (torch.abs(di_j - dj) / dj) < rth
    visible = consistent & validj
    if validi is not None:
        visible = visible & validi
    return visible


def batch_project_and_check_depth(
    kpi_3d_i: torch.Tensor,
    depthj: torch.Tensor,
    Kj: torch.Tensor,
    T_itoj: torch.Tensor,
    validi: torch.Tensor,
    rth=0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Projects 3D points to image pixels coordinates, and run depth reprojection erorr check.
    Args:
        kpi: BxNx2 coordinates in pixels
        di: BxN, corresponding depths
        depthj: depth map of image j, BxHxW
        Kj: Bx3x3
        T_itoj: Bx4x4
        validi: BxN, bool
        rth: percentage of depth error acceptable

    Returns:
        pixels projections in image j, BxNx2
        visible: torch.Bool BxN, indicates if the point has a valib depth reprojection error.
    """
    kpi_3d_j = from_homogeneous(to_homogeneous(kpi_3d_i) @ T_itoj.transpose(-1, -2))
    kpi_j = from_homogeneous(kpi_3d_j @ Kj.transpose(-1, -2))
    di_j = kpi_3d_j[..., -1]

    dj, validj = sample_depth(kpi_j, depthj)

    consistent = (torch.abs(di_j - dj) / dj) < rth
    visible = validi & consistent & validj
    return kpi_j, visible


def batch_transform(kpi_3d_i: torch.Tensor, T_itoj: torch.Tensor) -> torch.Tensor:
    """
    Args:
        kpi_3d_i: BxNx3 coordinates in frame i
        T_itoj: Bx4x4
    """
    kpi_3d_j = from_homogeneous(to_homogeneous(kpi_3d_i) @ T_itoj.transpose(-1, -2))
    return kpi_3d_j


def attentive_indexing(kps, imside, thres=0.1):
    r"""kps: normalized keypoints x, y (N, 2)
    returns attentive index map(N, spatial_side, spatial_side)
    """
    nkps = kps.size(0)
    kps = kps.view(nkps, 1, 1, 2)

    eps = 1e-5

    grid = (
        torch.stack(
            list(
                reversed(
                    torch.meshgrid(
                        torch.linspace(-1, 1, imside), torch.linspace(-1, 1, imside)
                    )
                )
            )
        )
        .permute(1, 2, 0)
        .to(kps)
    )

    attmap = (grid.unsqueeze(0).repeat(nkps, 1, 1, 1) - kps).pow(2).sum(dim=3)
    attmap = (attmap + eps).pow(0.5)
    attmap = (thres - attmap).clamp(min=0).view(nkps, -1)
    attmap = attmap / (attmap.sum(dim=1, keepdim=True) + eps)
    attmap = attmap.view(nkps, imside, imside)

    return attmap


def apply_gaussian_kernel(corr, imside=256, sigma=10):
    side, side = corr.size()

    center = corr.max(dim=-1)[1]
    center_y = center // imside
    center_x = center % imside
    y = torch.arange(0, imside).float().to(corr)
    x = torch.arange(0, imside).float().to(corr)

    y = y.view(1, imside).repeat(center_y.size(-1), 1) - center_y.unsqueeze(1)
    x = x.view(1, imside).repeat(center_x.size(-1), 1) - center_x.unsqueeze(1)

    y = y.unsqueeze(2).repeat(1, 1, imside)
    x = x.unsqueeze(1).repeat(1, imside, 1)

    gauss_kernel = torch.exp(-(x.pow(2) + y.pow(2)) / (2 * sigma**2))
    filtered_corr = gauss_kernel * corr.view(-1, imside, imside)
    filtered_corr = filtered_corr.view(side, side)

    return filtered_corr


def transfer_kps_diff(correlation, src_kps, imside=256, debug=None):
    r"""Transfer keypoints by weighted average"""
    thres = 0.1
    # no batch
    # input unnormalized keypoints
    src_kps -= imside // 2
    src_kps = src_kps / (imside // 2)

    correlation = apply_gaussian_kernel(correlation, imside)

    grid_x = torch.linspace(-1, 1, imside).to(src_kps)
    grid_x = grid_x.view(1, -1).repeat(imside, 1).view(1, -1)

    grid_y = torch.linspace(-1, 1, imside).to(src_kps)
    grid_y = grid_y.view(-1, 1).repeat(1, imside).view(1, -1)

    pdf = F.softmax(correlation / 0.0001, dim=-1)
    prd_x = (pdf * grid_x).sum(dim=-1)
    prd_y = (pdf * grid_y).sum(dim=-1)

    # per-pair
    np = src_kps.size(0)
    prd_xy = torch.stack([prd_x, prd_y]).t()

    attmap = attentive_indexing(src_kps, imside, thres).view(np, -1)
    prd_kp = (prd_xy.unsqueeze(0) * attmap.unsqueeze(-1)).sum(dim=1)

    prd_kp = prd_kp * (imside // 2) + (imside // 2)

    return prd_kp


def MutualMatching(corr4d):
    # mutual matching
    batch_size, ch, fs1, fs2, fs3, fs4 = corr4d.size()

    corr4d_B = corr4d.view(batch_size, fs1 * fs2, fs3, fs4)  # [batch_idx,k_A,i_B,j_B]
    corr4d_A = corr4d.view(batch_size, fs1, fs2, fs3 * fs4)

    # get max
    corr4d_B_max, _ = torch.max(corr4d_B, dim=1, keepdim=True)
    corr4d_A_max, _ = torch.max(corr4d_A, dim=3, keepdim=True)

    eps = 1e-5
    corr4d_B = corr4d_B / (corr4d_B_max + eps)
    corr4d_A = corr4d_A / (corr4d_A_max + eps)

    corr4d_B = corr4d_B.view(batch_size, 1, fs1, fs2, fs3, fs4)
    corr4d_A = corr4d_A.view(batch_size, 1, fs1, fs2, fs3, fs4)

    corr4d = corr4d * (
        corr4d_A * corr4d_B
    )  # parenthesis are important for symmetric output

    return corr4d


def interpolate4d(tensor4d, size):
    bsz, h1, w1, h2, w2 = tensor4d.size()
    tensor4d = tensor4d.view(bsz, h1, w1, -1).permute(0, 3, 1, 2)
    tensor4d = F.interpolate(tensor4d, size, mode="bilinear", align_corners=True)
    tensor4d = tensor4d.view(bsz, h2, w2, -1).permute(0, 3, 1, 2)
    tensor4d = F.interpolate(tensor4d, size, mode="bilinear", align_corners=True)
    tensor4d = tensor4d.view(bsz, size[0], size[0], size[0], size[0])
    return tensor4d


def fast4d(corr, kernel, bias=None):
    r"""Optimized implementation of 4D convolution
    taken from https://github.com/juhongm999/chm/blob/main/model/base/chm.py"""
    bsz, ch, srch, srcw, trgh, trgw = corr.size()
    out_channels, _, kernel_size, kernel_size, kernel_size, kernel_size = kernel.size()
    psz = kernel_size // 2

    out_corr = torch.zeros((bsz, out_channels, srch, srcw, trgh, trgw)).cuda()
    corr = corr.transpose(1, 2).contiguous().view(bsz * srch, ch, srcw, trgh, trgw)

    for pidx, k3d in enumerate(kernel.permute(2, 0, 1, 3, 4, 5)):
        inter_corr = F.conv3d(corr, k3d, bias=None, stride=1, padding=psz)
        inter_corr = (
            inter_corr.view(bsz, srch, out_channels, srcw, trgh, trgw)
            .transpose(1, 2)
            .contiguous()
        )

        add_sid = max(psz - pidx, 0)
        add_fid = min(srch, srch + psz - pidx)
        slc_sid = max(pidx - psz, 0)
        slc_fid = min(srch, srch - psz + pidx)

        out_corr[:, :, add_sid:add_fid, :, :, :] += inter_corr[
            :, :, slc_sid:slc_fid, :, :, :
        ]

    if bias is not None:
        out_corr += bias.view(1, out_channels, 1, 1, 1, 1)

    return out_corr
