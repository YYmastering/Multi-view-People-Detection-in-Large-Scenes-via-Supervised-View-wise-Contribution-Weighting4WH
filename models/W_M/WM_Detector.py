import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from torchvision.models.vgg import vgg16

from models.resnet import resnet18


class PerspTransDetector(nn.Module):
    """2D, supervised-view and view-weighted detector for fixed cameras."""

    def __init__(self, dataset, args):
        super().__init__()
        self.args = args
        self.variant = args.variant
        self.num_cam = dataset.num_cam
        self.img_shape = list(dataset.img_shape)
        self.reducedgrid_shape = list(dataset.reducedgrid_shape)
        self.device_name = torch.device(args.device)
        self.upsample_shape = [
            int(size / dataset.img_reduce) for size in self.img_shape
        ]

        image_to_grid = self.get_imgcoord2worldgrid_matrices(
            dataset.base.intrinsic_matrices,
            dataset.base.extrinsic_matrices,
            dataset.base.worldgrid2worldcoord_mat,
        )
        image_reduce = np.asarray(self.img_shape) / np.asarray(
            self.upsample_shape
        )
        image_zoom = np.diag(np.append(image_reduce, 1.0))
        map_zoom = np.diag(
            np.append(np.ones(2) / dataset.grid_reduce, 1.0)
        )
        projection_matrices = np.stack([
            map_zoom @ image_to_grid[camera] @ image_zoom
            for camera in range(self.num_cam)
        ])
        self.register_buffer(
            "proj_mats", torch.from_numpy(projection_matrices).float(),
            persistent=False,
        )

        if args.arch == "vgg16":
            self.base = vgg16().features[:16]
            out_channels = 256
        elif args.arch == "resnet18":
            backbone = resnet18(
                pretrained=not args.no_pretrained_backbone,
                replace_stride_with_dilation=[False, True, True],
            )
            self.base = nn.Sequential(*list(backbone.children())[:-2])
            out_channels = 512
        else:
            raise ValueError("architecture must be vgg16 or resnet18")

        self.img_classifier = nn.Sequential(
            nn.Conv2d(out_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1, bias=False),
        )
        self.single_view_classifier = nn.Sequential(
            nn.Conv2d(out_channels, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1, bias=False),
        )
        self.map_classifier = nn.Sequential(
            nn.Conv2d(out_channels, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, 3, padding=1, bias=False),
        )
        self.GP_view_CNN = nn.Sequential(
            nn.Conv2d(1, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1, bias=False),
        )

        self._frozen_modules = []
        if args.fix_2D:
            self._freeze(self.base, self.img_classifier)
        if args.fix_svp:
            self._freeze(self.single_view_classifier)
        if args.fix_weight:
            self._freeze(self.GP_view_CNN)
        self.to(self.device_name)

    def _freeze(self, *modules):
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad = False
            self._frozen_modules.append(module)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            for module in self._frozen_modules:
                module.eval()
        return self

    def forward(self, images):
        batch_size, num_views, _, _, _ = images.shape
        if num_views != self.num_cam:
            raise ValueError(
                f"expected {self.num_cam} cameras, received {num_views}"
            )
        images = images.to(self.device_name, non_blocking=True)

        image_results = []
        world_features = []
        for camera in range(self.num_cam):
            image_feature = self.base(images[:, camera])
            image_feature = F.interpolate(
                image_feature,
                size=self.upsample_shape,
                mode="bilinear",
                align_corners=False,
            )
            image_results.append(self.img_classifier(image_feature))
            if self.variant != "2D":
                projection = self.proj_mats[camera].unsqueeze(0).expand(
                    batch_size, -1, -1
                )
                world_features.append(
                    kornia.geometry.warp_perspective(
                        image_feature, projection, self.reducedgrid_shape,
                        align_corners=True,
                    )
                )

        image_results = torch.stack(image_results, dim=1).flatten(0, 1)
        if self.variant == "2D":
            return image_results

        world_features = torch.stack(world_features, dim=1)
        _, _, feature_channels, grid_h, grid_w = world_features.shape
        flat_world_features = world_features.reshape(
            batch_size * self.num_cam, feature_channels, grid_h, grid_w
        )
        view_results = self.single_view_classifier(flat_world_features)
        view_masks = (
            torch.linalg.vector_norm(world_features, dim=2, keepdim=True) > 0
        ).to(world_features.dtype)
        if self.variant == "2D_SVP":
            return image_results, view_results, view_masks.flatten(0, 1)
        if self.variant != "2D_SVP_VCW":
            raise ValueError(f"unsupported variant: {self.variant}")

        weight_logits = self.GP_view_CNN(view_results).reshape(
            batch_size, self.num_cam, 1, grid_h, grid_w
        )
        weight_logits = weight_logits - weight_logits.amax(
            dim=1, keepdim=True
        )
        weights = torch.exp(weight_logits) * view_masks
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-12)
        fused_features = (weights * world_features).sum(dim=1)
        ground_result = self.map_classifier(fused_features)
        return (
            image_results,
            view_results,
            ground_result,
            view_masks.flatten(0, 1),
        )

    @staticmethod
    def get_imgcoord2worldgrid_matrices(
        intrinsic_matrices, extrinsic_matrices, worldgrid2worldcoord_mat
    ):
        matrices = {}
        permutation = np.asarray(
            [[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64
        )
        for camera in range(len(intrinsic_matrices)):
            world_to_image = intrinsic_matrices[camera] @ np.delete(
                extrinsic_matrices[camera], 2, axis=1
            )
            grid_to_image = world_to_image @ worldgrid2worldcoord_mat
            matrices[camera] = permutation @ np.linalg.inv(grid_to_image)
        return matrices
