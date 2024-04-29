import os
from dataclasses import dataclass, field
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce


import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.misc import cleanup, get_device
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
from threestudio.utils.batched_geometry_utils import *

from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)

# visualization
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch

# debugging
from torchvision.utils import save_image
import torchvision.transforms as T
import random
import cv2


@threestudio.register("corr-mvdream-system")
class MVDreamSystem(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        visualize_samples: bool = False

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.guidance.requires_grad_(False)
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()

    def on_load_checkpoint(self, checkpoint):
        for k in list(checkpoint["state_dict"].keys()):
            if k.startswith("guidance."):
                return
        guidance_state_dict = {
            "guidance." + k: v for (k, v) in self.guidance.state_dict().items()
        }
        checkpoint["state_dict"] = {**checkpoint["state_dict"], **guidance_state_dict}
        return

    def on_save_checkpoint(self, checkpoint):
        for k in list(checkpoint["state_dict"].keys()):
            if k.startswith("guidance."):
                checkpoint["state_dict"].pop(k)
        return

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return self.renderer(**batch)

    # NOTE: Create another version of the batch with azimuth-moved camera positions, to find easier correspondences.
    def copy_batch_azimuth_perturbation(self, batch, perturbation):

        camera_distances = batch["camera_distances"]
        elevation = batch["elevation"] / 180 * math.pi
        azimuth_deg = batch["azimuth"] + perturbation
        azimuth = azimuth_deg / 180 * math.pi

        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        light_direction: Float[Tensor, "B 3"] = F.normalize(
            camera_positions + batch["light_random"],
            dim=-1,
        )
        # get light position by scaling light direction by light distance
        light_positions: Float[Tensor, "B 3"] = (
            light_direction * batch["light_distances"][:, None]
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions).to(
            camera_positions
        )
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = (
            torch.as_tensor([0, 0, 1], dtype=torch.float32)[None, :]
            .repeat(camera_distances.size(0), 1)
            .to(camera_positions)
        )

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # Importance note: the returned rays_d MUST be normalized!
        rays_o, rays_d = get_rays(batch["ray_directions"], c2w, keepdim=True)

        return {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "c2w": c2w,
            "light_positions": light_positions,
            "azimuth": azimuth_deg,
            "camera_positions": camera_positions,
            "elevation": batch["elevation"],
            "camera_distances": batch["camera_distances"],
            "height": batch["height"],
            "width": batch["width"],
            "fovy": batch["fovy"],
            "focal_length": batch["focal_length"],
            "mvp_mtx": None,
        }

    def training_step(self, batch, batch_idx):
        assert self.cfg.loss.use_corr_loss, "Correspondence loss must be enabled"

        start_time = time.time_ns()
        num_imgs = bsize = batch["rays_o"].size(0)
        img_size = batch["height"]

        # OpenGL standard to OpenCV standard
        base_c2w = batch["c2w"] @ torch.diag(torch.FloatTensor([1, -1, -1, 1])).cuda()

        out = self(batch)
        if (
            self.true_global_step >= self.cfg.loss.use_corr_after
            and self.true_global_step < self.cfg.loss.use_corr_until
            and self.true_global_step % 2 == 0
        ):
            with torch.no_grad():
                batch_corr = self.copy_batch_azimuth_perturbation(
                    batch,
                    perturbation=random.randint(
                        self.cfg.loss.azimuth_perturbation_start,
                        self.cfg.loss.azimuth_perturbation_end,
                    ),
                )
                base_corr_c2w = (
                    batch_corr["c2w"]
                    @ torch.diag(torch.FloatTensor([1, -1, -1, 1])).cuda()
                )
                out_corr = self(batch_corr)

        if (
            self.true_global_step >= self.cfg.loss.use_corr_after
            and self.true_global_step < self.cfg.loss.use_corr_until
            and self.true_global_step % 2 == 0
        ):
            guidance_out = self.guidance(
                out["comp_rgb"], self.prompt_utils, timestep=0, **batch
            )
        else:
            guidance_out = self.guidance(out["comp_rgb"], self.prompt_utils, **batch)

        if (
            self.true_global_step >= self.cfg.loss.use_corr_after
            and self.true_global_step < self.cfg.loss.use_corr_until
            and self.true_global_step % 2 == 0
        ):
            with torch.no_grad():
                guidance_out_corr = self.guidance(
                    out_corr["comp_rgb"],
                    self.prompt_utils,
                    timestep=guidance_out["t"],
                    noise=guidance_out["noise"],
                    **batch_corr,
                )

        loss = 0.0

        if (
            self.true_global_step >= self.cfg.loss.use_corr_after
            and self.true_global_step < self.cfg.loss.use_corr_until
            and self.true_global_step % 2 == 0
        ):
            pass
        else:
            for name, value in guidance_out.items():
                if name.startswith("loss_"):
                    self.log(f"train/{name}", value)
                    loss += value * self.C(
                        self.cfg.loss[name.replace("loss_", "lambda_")]
                    )

        ######################################################################
        # Corr loss implementation                                           #
        ######################################################################
        if (
            self.true_global_step >= self.cfg.loss.use_corr_after
            and self.true_global_step < self.cfg.loss.use_corr_until
            and self.true_global_step % 2 == 0
        ):

            corres_loss = 0

            if self.cfg.loss.visualize:
                with torch.no_grad():
                    fig_size = 5
                    fig, axes = plt.subplots(
                        num_imgs, 6, figsize=(fig_size * 6, fig_size * num_imgs)
                    )
                    tensor_to_PIL = T.ToPILImage()

                    for i in range(num_imgs):
                        for j in range(3):
                            axes[i][j * 2].imshow(
                                tensor_to_PIL(out["comp_rgb"][i].permute(2, 0, 1)),
                                resample=2,
                                alpha=0.3 if j > 0 else 1,
                            )
                            axes[i][j * 2 + 1].imshow(
                                tensor_to_PIL(out_corr["comp_rgb"][i].permute(2, 0, 1)),
                                resample=2,
                                alpha=0.3 if j > 0 else 1,
                            )
                            axes[i][j * 2].axis("off")
                            axes[i][j * 2 + 1].axis("off")

            for i in range(num_imgs):
                with torch.no_grad():
                    # intrinsics := using fovy
                    focal = batch["focal_length"][i]
                    zeros = torch.zeros_like(focal)
                    ones = torch.ones_like(focal)

                    intrinsics = (
                        torch.Tensor(
                            ([focal] + [zeros] * 3 + [focal] + [zeros] * 3 + [ones])
                        )
                        .view(3, 3)
                        .to(focal)
                    )
                    intrinsics[:2, -1] = batch["height"] / 2

                    # Filter out whose neighbours are of low opacity i.e., near edges
                    src_opacity = (out["opacity"][i]).permute(2, 0, 1)
                    trg_opacity = (out_corr["opacity"][i]).permute(2, 0, 1)

                    kernel_size = self.cfg.loss.edge_ksz

                    src_opacity = F.avg_pool2d(
                        src_opacity,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=kernel_size // 2,
                    )
                    trg_opacity = F.avg_pool2d(
                        trg_opacity,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=kernel_size // 2,
                    )

                    sparse_indices = (src_opacity.squeeze(0) > 0.99).nonzero()

                    if len(sparse_indices) > self.cfg.loss.max_corr:
                        random_values = torch.randperm(
                            len(sparse_indices), device=sparse_indices.device
                        )[: self.cfg.loss.max_corr]
                        sparse_indices = sparse_indices[random_values]

                    cossims = []
                    for feat_id in [4, 7]:

                        src_feat = guidance_out["unet_features"][-feat_id][:num_imgs][i]
                        trg_feat = guidance_out_corr["unet_features"][-feat_id][
                            :num_imgs
                        ][i]

                        src_feat = nn.Upsample(
                            size=(img_size, img_size), mode="bicubic"
                        )(src_feat.unsqueeze(0))
                        trg_feat = nn.Upsample(
                            size=(img_size, img_size), mode="bicubic"
                        )(trg_feat.unsqueeze(0))

                        src_feat = src_feat / src_feat.norm(p=2, dim=1, keepdim=True)
                        trg_feat = trg_feat / trg_feat.norm(p=2, dim=1, keepdim=True)

                        corr = torch.einsum(
                            "bcij,bckl->bijkl", src_feat, trg_feat
                        ).unsqueeze(0)
                        corr = MutualMatching(corr)

                        kernel = torch.ones(1, 1, 3, 3, 3, 3).half().to(corr) / (3**4)
                        corr = fast4d(corr, kernel).squeeze(0).squeeze(0)
                        corr = rearrange(corr, "i j k l -> (i j) (k l)")
                        cossims.append(corr)

                    cossim = torch.stack(cossims, dim=0).mean(dim=0)

                    # Absolute nn
                    trg_indices = cossim[
                        (sparse_indices[:, 0] * img_size + sparse_indices[:, 1])
                    ].argmax(dim=-1)
                    trg_indices = torch.stack(
                        (trg_indices // img_size, trg_indices % img_size), dim=1
                    )

                    trg_indices_yx = trg_indices[:, [1, 0]]

                    # Calculate fundamental matrix
                    s2t = pose_inverse_4x4(base_corr_c2w[i]) @ base_c2w[i]
                    R = s2t[:3, :3]
                    t = s2t[:3, 3]
                    t_cross = torch.tensor(
                        [[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]]
                    ).to(R)

                    Fmat = (
                        intrinsics.inverse().t() @ (t_cross @ R) @ intrinsics.inverse()
                    )
                    sparse_indices_yx = sparse_indices[:, [1, 0]]

                    sparse_indices_yx_h = torch.cat(
                        (
                            sparse_indices_yx + 0.5,
                            torch.ones(sparse_indices_yx.size(0))
                            .unsqueeze(-1)
                            .to(sparse_indices_yx),
                        ),
                        dim=-1,
                    )

                    line_eq = Fmat @ (
                        sparse_indices_yx_h.t().float()
                    )  # shape 3xN, need to pad with ones
                    y0 = -line_eq[2] / line_eq[1]  # at x = 0
                    y1 = -(line_eq[2] + line_eq[0] * img_size) / line_eq[1]

                    # get points for epipolar line
                    p0 = torch.stack((torch.zeros_like(y0).to(y0), y0))
                    p1 = torch.stack((torch.ones_like(y1).to(y1) * img_size, y1))
                    l2 = torch.sum(
                        (p0 - p1) ** 2, dim=0
                    )  # distance between the two points.

                    # projection
                    t = torch.clamp(
                        torch.sum((trg_indices_yx.t() + 0.5 - p0) * (p1 - p0), dim=0)
                        / l2,
                        min=0,
                        max=1,
                    )

                    trg_indices_yx_proj = (p0 + t * (p1 - p0)).t()

                    proj_distance = torch.sum(
                        (trg_indices_yx_proj - trg_indices_yx - 0.5) ** 2, dim=1
                    )  # squared distance
                    # Filter those projected too much:
                    projected_close = proj_distance < self.cfg.loss.epipolar_thres

                    sparse_indices_yx = sparse_indices_yx[projected_close]
                    trg_indices_yx_proj = trg_indices_yx_proj[projected_close]

                    # the other end should be within the image
                    within_bounds = (
                        (trg_indices_yx_proj[:, 0] > 0)
                        & (trg_indices_yx_proj[:, 0] < img_size)
                        & (trg_indices_yx_proj[:, 1] > 0)
                        & (trg_indices_yx_proj[:, 1] < img_size)
                    )

                    sparse_indices_yx = sparse_indices_yx[within_bounds]
                    trg_indices_yx_proj = trg_indices_yx_proj[within_bounds]

                    # the other end should also be within the object (with sufficiently high opacity)
                    trg_is_opaque = (
                        trg_opacity[0][
                            trg_indices_yx_proj[:, 1].int(),
                            trg_indices_yx_proj[:, 0].int(),
                        ]
                        > 0.99
                    )

                    sparse_indices_yx = sparse_indices_yx[trg_is_opaque]
                    trg_indices_yx_proj = trg_indices_yx_proj[trg_is_opaque]

                    corr_confidence = cossim[
                        sparse_indices_yx[:, 1] * img_size + sparse_indices_yx[:, 0],
                        trg_indices_yx_proj[:, 1].int() * img_size
                        + trg_indices_yx_proj[:, 0].int(),
                    ]

                # Depth of images d_A, d_B := rendered depths
                d_A = out["depth"][i][sparse_indices_yx[:, 1], sparse_indices_yx[:, 0]]

                # 1. Source to target
                pts_self_repr_in_other, depth_self_repr_in_other = (
                    batch_project_to_other_img(
                        sparse_indices_yx.float() + 0.5,
                        di=d_A,
                        Ki=intrinsics,
                        Kj=intrinsics,
                        src_c2w=base_c2w[i],
                        trg_c2w=base_corr_c2w[i],
                        return_depth=True,
                    )
                )

                diff_self = pts_self_repr_in_other - trg_indices_yx_proj
                corres_loss_tensor = nn.functional.huber_loss(
                    diff_self, torch.zeros_like(diff_self), reduction="none"
                ) * corr_confidence.unsqueeze(-1)
                corres_loss += corres_loss_tensor.sum() / (
                    corres_loss_tensor.nelement() + 1e-6
                )
                self.log(
                    "train/num_corr_loss_{}".format(i), corres_loss_tensor.nelement()
                )

                # 1. Colour the 3rd and 4th column images based on the diff values

                if self.cfg.loss.visualize:
                    with torch.no_grad():
                        # prolly need torch tensor to numpy conversion
                        axes[i][2].scatter(
                            sparse_indices_yx[:, 0].cpu().numpy(),
                            sparse_indices_yx[:, 1].cpu().numpy(),
                            c=np.abs((diff_self**2).sum(-1).cpu().numpy()),
                        )
                        axes[i][3].scatter(
                            trg_indices_yx_proj[:, 0].cpu().numpy(),
                            trg_indices_yx_proj[:, 1].cpu().numpy(),
                            c=np.abs((diff_self**2).sum(-1).cpu().numpy()),
                        )

                        topk_indices = torch.sum(diff_self**2, dim=1).topk(
                            k=sparse_indices_yx.size(0) // 5
                        )[1]

                        sparse_indices_yx = sparse_indices_yx[topk_indices]
                        trg_indices_yx_proj = trg_indices_yx_proj[topk_indices]

                        colors = ["red", "blue", "green", "yellow", "pink", "cyan"]
                        for src_c, trg_c in zip(sparse_indices_yx, trg_indices_yx_proj):
                            src_c = src_c.cpu().numpy()
                            trg_c = trg_c.cpu().numpy()

                            con = ConnectionPatch(
                                xyA=src_c,
                                xyB=trg_c,
                                coordsA=axes[i][4].transData,
                                coordsB=axes[i][5].transData,
                                axesA=axes[i][4],
                                axesB=axes[i][5],
                                color=random.choice(colors),
                            )
                            fig.add_artist(con)
            if self.cfg.loss.visualize:
                if self.true_global_step % 100 == 0:
                    plt.savefig(
                        "test_images/{}_{}_viz.png".format(
                            self.cfg.loss.visualize_name, self.true_global_step
                        )
                    )
                plt.close()

            self.log("train/corr_loss", corres_loss)
            loss += corres_loss * self.C(self.cfg.loss.lambda_corr)
        ######################################################################
        # Corr loss implementation end                                       #
        ######################################################################

        if self.C(self.cfg.loss.lambda_orient) > 0:
            if "normal" not in out:
                raise ValueError(
                    "Normal is required for orientation loss, no normal is found in the output."
                )
            loss_orient = (
                out["weights"].detach()
                * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
            ).sum() / (out["opacity"] > 0).sum()
            self.log("train/loss_orient", loss_orient)
            loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
