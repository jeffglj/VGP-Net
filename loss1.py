import math

import torch
import torch.nn.functional as F


def batch_episym(x1, x2, F):
    batch_size, num_pts = x1.shape[0], x1.shape[1]
    x1 = torch.cat([x1, x1.new_ones(batch_size, num_pts, 1)], dim=-1).reshape(batch_size, num_pts, 3, 1)
    x2 = torch.cat([x2, x2.new_ones(batch_size, num_pts, 1)], dim=-1).reshape(batch_size, num_pts, 3, 1)
    F = F.reshape(-1, 1, 3, 3).repeat(1, num_pts, 1, 1)
    x2Fx1 = torch.matmul(x2.transpose(2, 3), torch.matmul(F, x1)).reshape(batch_size, num_pts)
    Fx1 = torch.matmul(F, x1).reshape(batch_size, num_pts, 3)
    Ftx2 = torch.matmul(F.transpose(2, 3), x2).reshape(batch_size, num_pts, 3)
    ys = x2Fx1 ** 2 * (
        1.0 / (Fx1[:, :, 0] ** 2 + Fx1[:, :, 1] ** 2 + 1e-15)
        + 1.0 / (Ftx2[:, :, 0] ** 2 + Ftx2[:, :, 1] ** 2 + 1e-15)
    )
    return ys


def pair_matrix(value, name):
    if value.ndim == 3 and value.shape[-1] == 1:
        value = value.squeeze(-1)
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError(f"{name} must have nonempty shape [B,N], got {tuple(value.shape)}")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} contains NaN/Inf; evaluation stopped, not filtered")
    return value


@torch.no_grad()
def prf_from_masks(predicted, positive):
    if predicted.dtype != torch.bool or positive.dtype != torch.bool:
        raise TypeError("P/R/F masks must be boolean")
    if predicted.shape != positive.shape or predicted.ndim != 2:
        raise ValueError("P/R/F masks must have the same [B,N] shape")
    true_pos = (predicted & positive).sum(dim=1).double()
    predicted_count = predicted.sum(dim=1).double()
    positive_count = positive.sum(dim=1).double()
    precision = true_pos / predicted_count.clamp_min(1)
    recall = true_pos / positive_count.clamp_min(1)
    f_score = 2.0 * true_pos / (predicted_count + positive_count).clamp_min(1)
    return precision, recall, f_score


class MatchLoss(object):
    def __init__(self, config):
        self.loss_essential = config.loss_essential
        self.loss_classif = config.loss_classif
        self.use_fundamental = config.use_fundamental
        self.obj_geod_th = config.obj_geod_th
        self.geo_loss_margin = config.geo_loss_margin
        self.loss_essential_init_iter = config.loss_essential_init_iter

        self.prf_mode = getattr(config, "prf_mode", "logits")
        self.prf_logit_head = int(getattr(config, "prf_logit_head", 1))
        self.prf_logit_threshold = 1.5
        if self.prf_mode not in ("geometry", "logits"):
            raise ValueError("prf_mode must be 'geometry' or 'logits'")
        if self.prf_logit_head < 0:
            raise ValueError("prf_logit_head must be nonnegative")
        if not math.isfinite(self.prf_logit_threshold):
            raise ValueError("prf_logit_threshold must be finite")
        if not math.isfinite(self.obj_geod_th) or self.obj_geod_th <= 0:
            raise ValueError("obj_geod_th must be finite and positive")

    def weight_estimation(self, gt_geod_d, is_pos, ones):
        dis = torch.abs(gt_geod_d - self.obj_geod_th) / self.obj_geod_th
        weight_p = torch.exp(-dis)
        weight_p = weight_p * is_pos
        weight_n = ones
        weight_n = weight_n * (1 - is_pos)
        weight = weight_p + weight_n
        return weight

    @torch.no_grad()
    def full_size_logits(self, data, logits):
        if not isinstance(logits, (list, tuple)) or self.prf_logit_head >= len(logits):
            raise ValueError("prf_logit_head is not present in the model's logit list")
        scores = pair_matrix(logits[self.prf_logit_head], "evaluation logits")
        target = pair_matrix(data["ys"], "ground-truth residuals")
        if scores.shape != target.shape:
            raise ValueError(
                f"Head {self.prf_logit_head} has shape {tuple(scores.shape)}, but the "
                f"full-input labels have {tuple(target.shape)}. Use a full-size head "
                "(VGP-Net head 1), not a pruned subset."
            )
        return scores

    @torch.no_grad()
    def classification_metrics(self, data, logits, y_hat):
        target = pair_matrix(data["ys"], "ground-truth residuals")
        if self.prf_mode == "logits":
            predicted = self.full_size_logits(data, logits) > self.prf_logit_threshold
        else:
            residual = pair_matrix(y_hat, "predicted geometric residuals")
            if residual.shape != target.shape:
                raise ValueError("Predicted residuals must cover all input correspondences")
            predicted = residual < self.obj_geod_th
        return prf_from_masks(predicted, target < self.obj_geod_th)

    def run(self, global_step, data, logits, ys, e_hat, y_hat):
        xs, pts_virt = data["xs"], data["virtPts"]
        pts1_virts, pts2_virts = pts_virt[:, :, :2], pts_virt[:, :, 2:]
        geod = batch_episym(pts1_virts, pts2_virts, e_hat[-1])
        essential_loss = torch.min(
            geod, self.geo_loss_margin * geod.new_ones(geod.shape)
        )
        essential_loss = essential_loss.mean()
        L2_loss = 0
        classif_loss = 0

        with torch.no_grad():
            ones = torch.ones((xs.shape[0], 1)).to(xs.device)

        for i in range(len(logits)):
            gt_geod_d = ys[i]
            is_pos = (gt_geod_d < self.obj_geod_th).type(gt_geod_d.type())
            is_neg = (gt_geod_d >= self.obj_geod_th).type(gt_geod_d.type())

            with torch.no_grad():
                pos = torch.sum(is_pos, dim=-1, keepdim=True)
                pos_num = F.relu(pos - 1) + 1
                neg = torch.sum(is_neg, dim=-1, keepdim=True)
                neg_num = F.relu(neg - 1) + 1
                pos_w = neg_num / pos_num
                pos_w = torch.max(pos_w, ones)
                weight = self.weight_estimation(gt_geod_d, is_pos, ones)

            classif_loss += F.binary_cross_entropy_with_logits(
                weight * logits[i], is_pos, pos_weight=pos_w
            )

        precision, recall, _ = self.classification_metrics(data, logits, y_hat)

        loss = 0
        if self.loss_essential > 0 and global_step >= self.loss_essential_init_iter:
            loss += self.loss_essential * essential_loss
        if self.loss_classif > 0:
            loss += self.loss_classif * classif_loss

        return [
            loss,
            (self.loss_essential * essential_loss).item(),
            (self.loss_classif * classif_loss).item(),
            L2_loss,
            precision.mean().item(),
            recall.mean().item(),
        ]
