import hashlib
import json
import os
import tempfile
from collections import defaultdict

import numpy as np
from torchvision.datasets import VisionDataset


class WarehouseCocoSplit(VisionDataset):
    """Warehouse COCO split restricted to Camera 1-4 and people."""

    def __init__(self, root, split="train", camera_ids=(1, 2, 3, 4)):
        root = os.path.abspath(os.path.expanduser(root))
        super().__init__(root)
        self.root = root
        self.split = split
        self.split_root = os.path.join(root, split)
        self.annotation_path = os.path.join(
            self.split_root, "_annotations.coco.json"
        )
        self.camera_ids = tuple(camera_ids)
        self.camera_names = tuple(
            f"Camera_{camera_id:04d}" for camera_id in self.camera_ids
        )
        self.calibration_camera_names = tuple(
            f"Camera_{camera_id:02d}" for camera_id in self.camera_ids
        )

        self.__name__ = "Warehouse"
        self.dataset_name = "Warehouse"
        self.num_cam = len(self.camera_ids)
        self.img_shape = [1080, 1920]
        self.input_img_shape = [720, 1280]
        self.worldgrid_shape = [400, 400]
        self.world_grid_shape = self.worldgrid_shape
        self.indexing = "ij"
        self.grid_cell = 10.0  # centimetres per full-resolution grid cell
        self.default_map_sigma = 1.25  # 50 cm on a grid reduced by four
        self.default_nms_dist = 5.0
        self.default_eval_dist = 10.0

        # Grid input is (row, column), while calibration uses (world x, world y).
        self.worldgrid2worldcoord_mat = np.asarray(
            [[0.0, 10.0, -2000.0],
             [10.0, 0.0, -2000.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        with open(self._find_calibration_path(), "r") as calibration_file:
            self.calibration_data = json.load(calibration_file)
        calibration = [
            self.get_intrinsic_extrinsic_matrix(camera)
            for camera in range(self.num_cam)
        ]
        self.intrinsic_matrices = tuple(item[0] for item in calibration)
        self.extrinsic_matrices = tuple(item[1] for item in calibration)

        self.img_fpaths, self.world_gt, self.head_gt = self._load_split()
        self.frame_ids = sorted(self.world_gt)
        self.num_frame = len(self.frame_ids)
        self.num_frames = self.num_frame

        root_hash = hashlib.sha1(self.root.encode("utf-8")).hexdigest()[:10]
        safe_split = "".join(c if c.isalnum() else "_" for c in split)
        self.gt_fpath = os.path.join(
            tempfile.gettempdir(),
            f"vcw_warehouse_{root_hash}_{safe_split}_gt.txt",
        )
        self._write_ground_truth()

        print(
            f"Warehouse split {split}: {self.num_frame} complete person frames, "
            "Camera 1-4"
        )

    def _find_calibration_path(self):
        candidates = (
            os.path.join(self.root, "Warehouse_008", "calibration.json"),
            os.path.join(self.root, "calibration.json"),
        )
        for path in candidates:
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(f"Cannot find calibration.json under {self.root}")

    def _load_split(self):
        if not os.path.isfile(self.annotation_path):
            raise FileNotFoundError(
                f"Cannot find COCO annotations: {self.annotation_path}"
            )

        with open(self.annotation_path, "r") as annotation_file:
            coco = json.load(annotation_file)

        camera_to_index = {
            name: index for index, name in enumerate(self.camera_names)
        }
        category_names = {
            category["id"]: category["name"]
            for category in coco.get("categories", [])
        }
        image_metadata = {
            image["id"]: image for image in coco.get("images", [])
        }

        image_paths = {camera: {} for camera in range(self.num_cam)}
        for image in coco.get("images", []):
            camera_name = image.get("camera")
            frame_value = image.get("frame_index")
            if camera_name not in camera_to_index or frame_value is None:
                continue
            frame = int(frame_value)
            image_paths[camera_to_index[camera_name]][frame] = os.path.join(
                self.split_root, image["file_name"]
            )

        all_frames = set().union(*(paths.keys() for paths in image_paths.values()))
        complete_frames = {
            frame for frame in all_frames
            if all(frame in image_paths[cam] for cam in range(self.num_cam))
        }

        world_objects = defaultdict(dict)
        head_objects = {
            camera: defaultdict(dict) for camera in range(self.num_cam)
        }
        for annotation in coco.get("annotations", []):
            image = image_metadata.get(annotation.get("image_id"), {})
            camera_name = annotation.get("camera", image.get("camera"))
            frame_value = annotation.get(
                "frame_index", image.get("frame_index")
            )
            if camera_name not in camera_to_index or frame_value is None:
                continue
            frame = int(frame_value)
            if frame not in complete_frames:
                continue

            category_name = category_names.get(
                annotation.get("category_id"), ""
            )
            object_type = annotation.get("object_type", category_name)
            if str(object_type).lower() != "person":
                continue

            object_id = annotation.get("object_id")
            location = annotation.get("location_3d")
            bbox = annotation.get("bbox")
            if object_id is None or location is None or len(location) < 2:
                continue
            if bbox is None or len(bbox) < 4:
                continue
            x, y, width, height = map(float, bbox[:4])
            if width <= 0 or height <= 0:
                continue

            world_point = self.get_worldgrid_from_location(location)
            if not (
                0 <= world_point[0] < self.worldgrid_shape[0]
                and 0 <= world_point[1] < self.worldgrid_shape[1]
            ):
                continue

            object_id = int(object_id)
            camera = camera_to_index[camera_name]
            world_objects[frame][object_id] = world_point
            # The original baseline supervises the head centre in each image.
            head_x = np.clip(x + width / 2.0, 0, self.img_shape[1] - 1)
            head_y = np.clip(y, 0, self.img_shape[0] - 1)
            head_objects[camera][frame][object_id] = (head_x, head_y)

        frames = sorted(frame for frame in complete_frames if world_objects[frame])
        filtered_paths = {
            camera: {frame: image_paths[camera][frame] for frame in frames}
            for camera in range(self.num_cam)
        }
        world_gt = {
            frame: np.asarray(
                [point for _, point in sorted(world_objects[frame].items())],
                dtype=np.float32,
            ).reshape(-1, 2)
            for frame in frames
        }
        head_gt = {
            frame: {
                camera: np.asarray(
                    [point for _, point in sorted(
                        head_objects[camera][frame].items()
                    )],
                    dtype=np.float32,
                ).reshape(-1, 2)
                for camera in range(self.num_cam)
            }
            for frame in frames
        }
        return filtered_paths, world_gt, head_gt

    @staticmethod
    def get_worldgrid_from_location(location):
        x_meters, y_meters = map(float, location[:2])
        return np.asarray(
            [y_meters * 10.0 + 200.0, x_meters * 10.0 + 200.0],
            dtype=np.float32,
        )

    def get_intrinsic_extrinsic_matrix(self, camera):
        sensor_id = self.calibration_camera_names[camera]
        sensor = next(
            (
                item for item in self.calibration_data["sensors"]
                if item.get("type", "").lower() == "camera"
                and item.get("id") == sensor_id
            ),
            None,
        )
        if sensor is None:
            raise ValueError(f"Calibration does not contain {sensor_id}")

        intrinsic = np.asarray(sensor["intrinsicMatrix"], dtype=np.float64)
        extrinsic = np.asarray(sensor["extrinsicMatrix"], dtype=np.float64)
        if intrinsic.shape != (3, 3) or extrinsic.shape != (3, 4):
            raise ValueError(f"Invalid calibration shape for {sensor_id}")
        extrinsic = extrinsic.copy()
        # Warehouse translations are metres; this ground plane is centimetres.
        extrinsic[:, 3] *= 100.0
        return intrinsic, extrinsic

    def get_image_fpaths(self, frame_range=None):
        if frame_range is None:
            return self.img_fpaths
        selected = set(frame_range)
        return {
            camera: {
                frame: path for frame, path in paths.items()
                if frame in selected
            }
            for camera, paths in self.img_fpaths.items()
        }

    get_image_paths = get_image_fpaths

    def _write_ground_truth(self):
        rows = [
            [frame, int(round(row)), int(round(column))]
            for frame in self.frame_ids
            for row, column in self.world_gt[frame]
        ]
        ground_truth = np.asarray(rows, dtype=np.int64).reshape(-1, 3)
        np.savetxt(self.gt_fpath, ground_truth, fmt="%d")
