import numpy as np
import torch
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.transforms import ToTensor

from utils import gaussian_blur_detecting


class frameDataset(VisionDataset):
    """Lazily generates Warehouse heatmaps without caching tens of GB."""

    def __init__(
        self,
        base,
        _transform=ToTensor(),
        target_transform=ToTensor(),
        grid_reduce=4,
        img_reduce=4,
        facofmaxgt=100,
        map_sigma=None,
        img_sigma=5,
    ):
        super().__init__(base.root, transform=_transform,
                         target_transform=target_transform)
        self.base = base
        self.root = base.root
        self.num_cam = base.num_cam
        self.frame_ids = list(base.frame_ids)
        self.num_frame = len(self.frame_ids)
        self.img_shape = list(base.img_shape)
        self.input_img_shape = list(base.input_img_shape)
        self.worldgrid_shape = list(base.worldgrid_shape)
        self.grid_reduce = int(grid_reduce)
        if self.grid_reduce <= 0 or any(
            size % self.grid_reduce for size in self.worldgrid_shape
        ):
            raise ValueError("grid_reduce must evenly divide the world grid")
        if img_reduce <= 0 or any(
            size % int(img_reduce) for size in self.input_img_shape
        ):
            raise ValueError("img_reduce must evenly divide the resized image")
        self.reducedgrid_shape = [
            int(size / self.grid_reduce) for size in self.worldgrid_shape
        ]
        self.upsample_shape = [
            int(size / img_reduce) for size in self.input_img_shape
        ]
        self.img_reduce = self.img_shape[0] / self.upsample_shape[0]
        self.facofmaxgt = float(facofmaxgt)
        self.map_sigma = (
            5.0 / self.grid_reduce
            if map_sigma is None else float(map_sigma)
        )
        self.img_sigma = float(img_sigma)
        self.img_fpaths = base.get_image_fpaths(self.frame_ids)
        self.gt_fpath = base.gt_fpath

    def __getitem__(self, index):
        frame = self.frame_ids[index]
        images = []
        for camera in range(self.num_cam):
            with Image.open(self.img_fpaths[camera][frame]) as image:
                image = image.convert("RGB")
                if self.transform is not None:
                    image = self.transform(image)
            images.append(image)
        images = torch.stack(images)

        ground_map = np.zeros(self.reducedgrid_shape, dtype=np.float32)
        for row, column in self.base.world_gt[frame]:
            center = (
                float(column) / self.grid_reduce,
                float(row) / self.grid_reduce,
            )
            gaussian_blur_detecting.draw_umich_gaussian(
                ground_map, center, sigma=self.map_sigma
            )
        ground_map *= self.facofmaxgt
        ground_map = torch.from_numpy(ground_map).unsqueeze(0)

        image_maps = []
        scale_x = self.upsample_shape[1] / self.img_shape[1]
        scale_y = self.upsample_shape[0] / self.img_shape[0]
        for camera in range(self.num_cam):
            image_map = np.zeros(self.upsample_shape, dtype=np.float32)
            for head_x, head_y in self.base.head_gt[frame][camera]:
                gaussian_blur_detecting.draw_umich_gaussian(
                    image_map,
                    (float(head_x) * scale_x, float(head_y) * scale_y),
                    sigma=self.img_sigma,
                )
            image_map *= self.facofmaxgt
            image_maps.append(torch.from_numpy(image_map).unsqueeze(0))

        return images, ground_map.float(), torch.stack(image_maps).float(), frame

    def __len__(self):
        return len(self.frame_ids)
