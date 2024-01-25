import math
from dataclasses import dataclass
from typing import Optional

import torch
from botorch.models.transforms.outcome import OutcomeTransform
from scipy import stats
from torch import distributions, nn

from cortex.model.branch import BranchNodeOutput
from cortex.model.leaf import LeafNode, LeafNodeOutput
from cortex.utils import check_scale
from cortex.metrics import spearman_rho


def diag_gaussian_nll(loc, scale, targets):
    dist = distributions.Normal(loc, scale)
    return -1.0 * dist.log_prob(targets).mean()


def diag_gaussian_cumulant(canon_param):
    res = -1.0 * canon_param[0].pow(2) / (4 * canon_param[1]) - 0.5 * (-2.0 * canon_param[1]).log()
    return res


def diag_natural_gaussian_nll(canon_param, targets):
    suff_stat = torch.stack([targets, targets.pow(2)])
    cumulant = diag_gaussian_cumulant(canon_param)
    underlying_measure = 1 / math.sqrt(2 * math.pi)
    log_likelihood = (
        math.log(underlying_measure) + (canon_param * suff_stat).sum(0) - cumulant
    ).mean()
    return -1.0 * log_likelihood


def diag_natural_gaussian_kl_divergence(canon_param_p, canon_param_q):
    # https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter8.pdf
    var_p = -0.5 / canon_param_p[1]
    mean_p = canon_param_p[0] * var_p.clamp_min(1e-4)

    exp_suff_stat = torch.stack(
        [
            mean_p,
            var_p + mean_p.pow(2),
        ]
    )

    term_1 = ((canon_param_p - canon_param_q) * exp_suff_stat).sum(0)
    term_2 = -1.0 * diag_gaussian_cumulant(canon_param_p)
    term_3 = diag_gaussian_cumulant(canon_param_q)
    return term_1 + term_2 + term_3


def inv_softplus(x):
    return x + torch.log(-torch.expm1(-x))


@dataclass
class RegressorLeafOutput(LeafNodeOutput):
    loc: torch.Tensor
    scale: torch.Tensor
    canon_param: Optional[torch.Tensor] = None


# class RegressorLeaf(LeafNode):
#     def __init__(
#         self,
#         in_dim: int,
#         out_dim: int,
#         branch_key: str,
#         log_scale_min: float = -8.0,
#         log_scale_max: float = 8.0,
#         num_layers: int = 0,
#         outcome_transform: Optional[OutcomeTransform] = None,
#         scale_bounds_loss_coeff: Optional[float] = None,
#         label_smoothing: float = 0.0,
#         nominal_label_var: float = 0.25**2,
#         root_key: Optional[str] = None,
#     ) -> None:
#         super().__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         assert log_scale_max > log_scale_min
#         # self.log_scale_min = log_scale_min
#         # self.log_scale_max = log_scale_max
#         self.register_parameter("log_scale_min", nn.Parameter(torch.tensor((log_scale_min))))
#         self.register_parameter("log_scale_max", nn.Parameter(torch.tensor((log_scale_max))))
#         self.branch_key = branch_key

#         encoder_modules = []
#         if num_layers >= 1:
#             for _ in range(num_layers):
#                 encoder_modules.extend(
#                     [
#                         nn.Linear(in_dim, in_dim),
#                         nn.ReLU(inplace=True),
#                     ]
#                 )
#         encoder_modules.append(nn.Linear(in_dim, out_dim * 2))
#         self.encoder = nn.Sequential(*encoder_modules)

#         self.loss_fn = diag_gaussian_nll
#         self.outcome_transform = outcome_transform
#         self.scale_bounds_loss_coeff = scale_bounds_loss_coeff
#         self.label_smoothing = label_smoothing
#         self.root_key = root_key
#         self.nominal_label_var = nominal_label_var

#     def forward(self, branch_outputs: SeqCNNBranchOutput) -> RegressorLeafOutput:
#         """
#         Return mean and std. dev. of diagonal Gaussian distribution
#         Args:
#             branch_outputs: {'branch_features': torch.Tensor, 'branch_mask': torch.Tensor, 'pooled_features': torch.Tensor}
#         Returns:
#             outputs: {'loc': torch.Tensor, 'scale': torch.Tensor}
#         """
#         res = self.encoder(branch_outputs.pooled_features)
#         loc, log_scale = res.chunk(2, dim=-1)
#         # constrain log_scale to [log_scale_min, log_scale_max]
#         log_scale_max = self.log_scale_max.detach()
#         log_scale_min = self.log_scale_min.detach()
#         log_scale = log_scale_max - nn.functional.softplus(log_scale_max - log_scale)
#         log_scale = log_scale_min + nn.functional.softplus(log_scale - log_scale_min)
#         # log_scale = torch.tanh(log_scale)
#         # log_scale = self.log_scale_min + 0.5 * (self.log_scale_max - self.log_scale_min) * log_scale
#         # scale = log_scale.exp()
#         scale = nn.functional.softplus(log_scale)

#         if isinstance(self.outcome_transform, OutcomeTransform):
#             self.outcome_transform.requires_grad_(False)
#             self.outcome_transform.eval()
#             # this is equivalent to transforming the training data
#             # BoTorch OutcomeTransform expects variance
#             loc, var = self.outcome_transform.untransform(loc, scale.pow(2))
#             scale = var.sqrt()

#         outputs = RegressorLeafOutput(
#             loc=loc,
#             scale=scale,
#         )
#         return outputs

#     def sample(self, pooled_features, num_samples):
#         outputs = self(pooled_features)
#         dist = distributions.Normal(outputs.loc, outputs.scale)
#         return dist.sample((num_samples,))

#     def _preprocess_targets(self, targets, device, dtype):
#         if not torch.is_tensor(targets):
#             targets = torch.tensor(targets)
#         targets = targets.to(device, dtype)
#         return targets

#     def loss(self, leaf_outputs: RegressorLeafOutput, targets, *args, **kwargs):
#         loc = leaf_outputs.loc
#         scale = leaf_outputs.scale
#         targets = self._preprocess_targets(targets, loc.device, loc.dtype)

#         if isinstance(self.outcome_transform, OutcomeTransform):
#             self.outcome_transform.eval()
#             loc, var = self.outcome_transform(loc, scale.pow(2))
#             scale = var.sqrt()
#             targets, _ = self.outcome_transform(targets)

#         nll = self.loss_fn(loc, scale, targets.detach())
#         if self.scale_bounds_loss_coeff is None:
#             scale_bounds_loss_term = 0.0
#         else:
#             scale_bounds_loss_term = self.scale_bounds_loss_coeff * (
#                 self.log_scale_max - self.log_scale_min
#             )
#         return nll + scale_bounds_loss_term

#     def evaluate(self, outputs: RegressorLeafOutput, targets):
#         loc = outputs.loc
#         scale = outputs.scale
#         targets = self._preprocess_targets(targets, loc.device, loc.dtype)
#         nrmse = torch.norm(loc - targets) / torch.norm(targets).clamp_min(1e-6)
#         spearman_rho = 0
#         for idx in range(targets.size(-1)):
#             spearman_rho += stats.spearmanr(
#                 targets[..., idx].cpu(), loc[..., idx].cpu()
#             ).correlation / targets.size(-1)
#         metrics = {
#             "nll": self.loss_fn(loc, scale, targets).item(),
#             "nrmse": nrmse.item(),
#             "s_rho": spearman_rho,
#         }
#         return metrics

#     def initialize(self) -> None:
#         """
#         initialize leaf weights
#         """
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)


class RegressorLeaf(LeafNode):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        branch_key: str,
        num_layers: int = 0,
        outcome_transform: Optional[OutcomeTransform] = None,
        label_smoothing: float = 0.0,
        nominal_label_var: float = 0.25**2,
        var_lb: float = 1e-4,
        root_key: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.branch_key = branch_key

        encoder_modules = []
        if num_layers >= 1:
            for _ in range(num_layers):
                encoder_modules.extend(
                    [
                        nn.Linear(in_dim, in_dim),
                        nn.ReLU(inplace=True),
                    ]
                )
        encoder_modules.append(nn.Linear(in_dim, out_dim * 2))
        self.encoder = nn.Sequential(*encoder_modules)

        self.loss_fn = diag_gaussian_nll
        self.outcome_transform = outcome_transform
        self.label_smoothing = label_smoothing
        self.root_key = root_key
        self.nominal_label_var = nominal_label_var
        self.var_lb = var_lb

    def forward(self, branch_outputs: BranchNodeOutput) -> RegressorLeafOutput:
        res = self.encoder(branch_outputs.pooled_features)
        return self.transform_output(res)
    
    def sample(self, pooled_features, num_samples):
        outputs = self(pooled_features)
        dist = distributions.Normal(outputs.loc, outputs.scale)
        return dist.sample((num_samples,))

    def _preprocess_targets(self, targets, device, dtype):
        if not torch.is_tensor(targets):
            targets = torch.tensor(targets)
        targets = targets.to(device, dtype)
        return targets

    def transform_output(self, nn_out: torch.Tensor) -> RegressorLeafOutput:
        """
        Return mean and std. dev. of diagonal Gaussian distribution
        Args:
            nn_out: torch.Tensor
        Returns:
            outputs: {'loc': torch.Tensor, 'scale': torch.Tensor}
        """
        res = nn_out.chunk(2, dim=-1)
        canon_param = torch.stack([res[0], -1.0 * nn.functional.softplus(res[1], beta=0.5)])

        var = (-0.5 / canon_param[1]).clamp_min(self.var_lb)
        loc = canon_param[0] * var
        scale = var.sqrt()

        if (
            isinstance(self.outcome_transform, OutcomeTransform)
            and self.outcome_transform._is_trained
        ):
            self.outcome_transform.eval()
            # this is equivalent to transforming the training data
            # BoTorch OutcomeTransform expects variance
            loc, var = self.outcome_transform.untransform(loc, var)
            # var = var.clamp_min(self.var_lb)
            scale = var.sqrt()

        outputs = RegressorLeafOutput(
            loc=loc,
            scale=scale,
            canon_param=canon_param,
        )
        return outputs

    def loss_from_canon_param(
        self, canon_param: torch.Tensor, targets: torch.Tensor, label_smoothing: float = 0.0
    ) -> torch.Tensor:
        device = canon_param.device
        dtype = canon_param.dtype
        targets = self._preprocess_targets(targets, device, dtype)

        if isinstance(self.outcome_transform, OutcomeTransform):
            self.outcome_transform.eval()
            targets, _ = self.outcome_transform(targets)
            targets = targets.detach()

        if label_smoothing > 0.0:
            # moment average
            # https://www.cs.toronto.edu/~cmaddis/pubs/aais.pdf
            standard_stats = torch.stack(
                [
                    torch.zeros_like(canon_param[0]),
                    torch.ones_like(canon_param[1]),
                ]
            )
            standard_stats.requires_grad_(False)

            label_stats = torch.stack(
                [
                    targets,
                    torch.full_like(targets, self.nominal_label_var),
                ]
            )
            label_stats = label_stats.detach()

            smoothed_mean = (
                label_smoothing * standard_stats[0] + (1.0 - label_smoothing) * label_stats[0]
            )
            mean_diff = standard_stats[0] - label_stats[0]
            cross_var_term = label_smoothing * (1.0 - label_smoothing) * mean_diff.pow(2)
            smoothed_var = (
                label_smoothing * standard_stats[1]
                + (1.0 - label_smoothing) * label_stats[1]
                + cross_var_term
            )

            # convert to natural parameters
            smoothed_params = torch.stack(
                [
                    smoothed_mean / smoothed_var,
                    -1.0 / (2 * smoothed_var),
                ]
            )

            return diag_natural_gaussian_kl_divergence(smoothed_params, canon_param).mean()
        else:
            return diag_natural_gaussian_nll(canon_param=canon_param, targets=targets)

    def loss(self, leaf_outputs: RegressorLeafOutput, root_outputs, targets, *args, **kwargs):
        if self.label_smoothing == "corrupt_frac" and hasattr(root_outputs, "corrupt_frac"):
            label_smoothing = root_outputs.corrupt_frac
        else:
            label_smoothing = self.label_smoothing

        canon_param = leaf_outputs.canon_param
        return self.loss_from_canon_param(canon_param, targets, label_smoothing)
    
    def evaluate(self, outputs: RegressorLeafOutput, targets):
        loc = outputs.loc
        scale = outputs.scale
        targets = self._preprocess_targets(targets, loc.device, loc.dtype)
        nrmse = torch.norm(loc - targets) / torch.norm(targets).clamp_min(1e-6)
        s_rho = spearman_rho(loc.cpu().numpy(), targets.cpu().numpy())
        metrics = {
            "nll": self.loss_fn(loc, scale, targets).item(),
            "nrmse": nrmse.item(),
            "s_rho": s_rho,
        }
        return metrics

    def initialize(self) -> None:
        """
        initialize leaf weights
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def format_regressor_ensemble_output(
    leaf_outputs: list[RegressorLeafOutput], task_key: str
) -> dict:
    res = {}
    loc = torch.stack([l_out.loc for l_out in leaf_outputs])
    scale = torch.stack([l_out.scale for l_out in leaf_outputs])
    check_scale(scale)
    res[f"{task_key}_mean"] = loc
    res[f"{task_key}_st_dev"] = scale
    return res
