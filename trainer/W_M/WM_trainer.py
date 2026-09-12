import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from evaluation.evaluate import evaluate
from utils.nms import nms


class PerspectiveTrainer:
    def __init__(self, model, args, logdir, denormalize=None):
        self.model = model
        self.args = args
        self.logdir = logdir
        self.num_cam = model.num_cam
        self.best_moda = float("-inf")

    @staticmethod
    def _flat_image_targets(image_targets):
        # [B, N, 1, H, W] -> sample-major [B*N, 1, H, W]
        return image_targets.flatten(0, 1)

    def _losses(self, outputs, ground_target, image_targets):
        image_result = outputs[0] if isinstance(outputs, tuple) else outputs
        image_target = self._flat_image_targets(image_targets).to(
            image_result.device, non_blocking=True
        )
        image_loss = F.mse_loss(image_result, image_target) / self.num_cam
        zero = image_loss.detach().new_zeros(())

        if self.args.variant == "2D":
            return image_loss, image_loss, zero, zero

        view_result, view_mask = outputs[1], outputs[-1]
        batch_size = ground_target.shape[0]
        ground_target = ground_target.to(view_result.device, non_blocking=True)
        view_target = ground_target.unsqueeze(1).expand(
            -1, self.num_cam, -1, -1, -1
        ).reshape_as(view_result)
        # Each SVP branch predicts its masked contribution to the global map.
        view_target = view_target * view_mask / self.num_cam
        view_loss = F.mse_loss(view_result, view_target)

        if self.args.variant == "2D_SVP":
            total = (
                self.args.weight_2D * image_loss.to(view_loss.device)
                + self.args.weight_svp * view_loss
            )
            return total, image_loss, view_loss, zero

        ground_result = outputs[2]
        fusion_loss = F.mse_loss(
            ground_result,
            ground_target.to(ground_result.device, non_blocking=True),
        )
        total = (
            fusion_loss
            + self.args.weight_2D * image_loss.to(fusion_loss.device)
            + self.args.weight_svp * view_loss.to(fusion_loss.device)
        )
        return total, image_loss, view_loss, fusion_loss

    def train(self, data_loader, epoch, optimizer, scheduler):
        self.model.train()
        sums = np.zeros(4, dtype=np.float64)
        start = time.time()

        for batch_index, (images, ground_target, image_targets, _) in enumerate(
            data_loader
        ):
            optimizer.zero_grad(set_to_none=True)
            outputs = self.model(images)
            loss_values = self._losses(outputs, ground_target, image_targets)
            loss_values[0].backward()
            optimizer.step()
            if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

            sums += np.asarray([value.item() for value in loss_values])
            if (
                batch_index == 0
                or (batch_index + 1) % self.args.log_interval == 0
                or batch_index + 1 == len(data_loader)
            ):
                averages = sums / (batch_index + 1)
                print(
                    f"Train epoch {epoch}, batch {batch_index + 1}/{len(data_loader)}, "
                    f"loss {averages[0]:.6f} "
                    f"(2D {averages[1]:.6f}, SVP {averages[2]:.6f}, "
                    f"fusion {averages[3]:.6f}), "
                    f"lr {optimizer.param_groups[0]['lr']:.8f}, "
                    f"time {time.time() - start:.1f}s"
                )

        if not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()
        return sums / max(len(data_loader), 1)

    def validate(self, data_loader, result_path=None, epoch=None):
        self.model.eval()
        sums = np.zeros(4, dtype=np.float64)
        candidates = []
        start = time.time()

        with torch.no_grad():
            for batch_index, (
                images, ground_target, image_targets, frames
            ) in enumerate(data_loader):
                outputs = self.model(images)
                loss_values = self._losses(
                    outputs, ground_target, image_targets
                )
                sums += np.asarray([value.item() for value in loss_values])

                if self.args.variant != "2D_SVP_VCW" or result_path is None:
                    continue
                ground_results = outputs[2].detach().cpu()
                for sample_index, frame in enumerate(frames.tolist()):
                    score_map = torch.relu(ground_results[sample_index, 0])
                    score_min, score_max = score_map.min(), score_map.max()
                    if (score_max - score_min).item() <= 1e-12:
                        continue
                    score_map = (score_map - score_min) / (score_max - score_min)
                    selected = score_map > self.args.cls_thres
                    positions = selected.nonzero(as_tuple=False)
                    if positions.numel() == 0:
                        continue
                    if data_loader.dataset.base.indexing == "xy":
                        positions = positions[:, [1, 0]]
                    scores = score_map[selected, None]
                    frame_column = torch.full_like(scores, float(frame))
                    candidates.append(torch.cat([
                        frame_column,
                        positions.float() * data_loader.dataset.grid_reduce,
                        scores,
                    ], dim=1))

        averages = sums / max(len(data_loader), 1)
        print(
            f"Validation epoch {epoch}, loss {averages[0]:.6f} "
            f"(2D {averages[1]:.6f}, SVP {averages[2]:.6f}, "
            f"fusion {averages[3]:.6f}), time {time.time() - start:.1f}s"
        )
        if self.args.variant != "2D_SVP_VCW" or result_path is None:
            return {"loss": float(averages[0])}

        all_candidates = (
            torch.cat(candidates, dim=0)
            if candidates else torch.empty((0, 4), dtype=torch.float32)
        )
        candidate_path = os.path.join(
            os.path.dirname(result_path), "all_res.txt"
        )
        np.savetxt(candidate_path, all_candidates.numpy(), fmt="%.8f")

        detections = []
        if all_candidates.numel():
            for frame in torch.unique(all_candidates[:, 0]):
                frame_candidates = all_candidates[
                    all_candidates[:, 0] == frame
                ]
                positions = frame_candidates[:, 1:3]
                scores = frame_candidates[:, 3]
                keep, count = nms(
                    positions, scores, self.args.nms_thres,
                    top_k=len(scores)
                )
                detections.append(torch.cat([
                    torch.full((count, 1), frame.item()),
                    positions[keep[:count]],
                ], dim=1))
        detections = (
            torch.cat(detections, dim=0).numpy()
            if detections else np.empty((0, 3), dtype=np.float32)
        )
        np.savetxt(result_path, detections, fmt="%d")

        recall, precision, moda, modp = evaluate(
            os.path.abspath(result_path),
            data_loader.dataset.gt_fpath,
            self.args.dist_thres,
            data_loader.dataset.base.__name__,
        )
        f1_score = 2 * precision * recall / (precision + recall + 1e-12)
        metrics = {
            "loss": float(averages[0]),
            "moda": float(moda),
            "modp": float(modp),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1_score),
        }
        print(
            f"MODA {moda:.1f}%, MODP {modp:.1f}%, precision {precision:.1f}%, "
            f"recall {recall:.1f}%, F1 {f1_score:.1f}%"
        )
        return metrics

    def save_checkpoint(self, path, epoch, optimizer, scheduler, metrics=None):
        checkpoint = {
            "epoch": epoch,
            "variant": self.args.variant,
            "model": self.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "metrics": metrics or {},
        }
        torch.save(checkpoint, path)
