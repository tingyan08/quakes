"""ARSGaussian's lidar-guided densification (Yao et al., arXiv:2412.18380, Sec. 3.1) on gsplat's DefaultStrategy.

Three changes, each switchable:
  pixel_gs     Eq. 4: the growth criterion averages each view's image-plane gradient weighted by the
               number of pixels the Gaussian covers in that view (Pixel-GS), not by view count.
  tangent_split  Eq. 5-6: inside the lidar area a split places the two children along the Gaussian's
               long axis projected onto the tangent plane of the nearest lidar point, not at random.
  lidar_prune  Eq. 3: every lidar_every steps, remove Gaussians inside the lidar area whose nearest lidar
               point is more than eps_plane away horizontally or eps_elev vertically, unless the lidar
               column holds them (a return above and, our addition, the Gaussian not below the lowest
               return by more than eps_elev - otherwise every under-ground floater would be kept).
Opacity pruning (O_i in Eq. 3) is the default strategy's prune_opa.
With all three off this is DefaultStrategy, plus max_gs (a cap on the number of Gaussians, for GPU memory).
"""
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from gsplat.strategy import DefaultStrategy
from gsplat.strategy.ops import _update_param_with_optimizer, duplicate, remove
from gsplat.utils import normalized_quat_to_rotmat


@dataclass
class LidarStrategy(DefaultStrategy):
    lidar_every: int = 300
    eps_plane: float = 2.5          # m
    eps_elev: float = 2.5           # m
    pixel_gs: bool = True
    tangent_split: bool = True
    lidar_prune: bool = True
    max_gs: int = 0                 # stop growing at this many Gaussians (0: no cap); not in the paper

    lidar: Any = None               # LidarGeom, set by the trainer

    def __getstate__(self):         # the trainer dumps its config to cfg.yml: leave the lidar out
        d = self.__dict__.copy()
        d["lidar"] = None
        return d

    # ---- Eq. 4: pixel-weighted gradient average
    def _update_state(self, params, state, info, packed=False):
        if not self.pixel_gs:
            return super()._update_state(params, state, info, packed=packed)
        grads = (info[self.key_for_gradient].absgrad if self.absgrad else info[self.key_for_gradient].grad).clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
        n = len(list(params.values())[0])
        for k in ("grad2d", "count"):
            if state[k] is None:
                state[k] = torch.zeros(n, device=grads.device)
        if self.refine_scale2d_stop_iter > 0 and state["radii"] is None:
            state["radii"] = torch.zeros(n, device=grads.device)
        if packed:
            gs_ids = info["gaussian_ids"]
            r = info["radii"].float()
        else:
            sel = (info["radii"] > 0.0).all(dim=-1)
            gs_ids = torch.where(sel)[1]
            grads = grads[sel]
            r = info["radii"][sel].float()
        m = (np.pi * r[:, 0] * r[:, 1]).clamp_min(1.0)              # pixels covered in this view
        state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1) * m)
        state["count"].index_add_(0, gs_ids, m)
        if self.refine_scale2d_stop_iter > 0:
            state["radii"][gs_ids] = torch.maximum(state["radii"][gs_ids],
                                                   r.max(dim=-1).values / float(max(info["width"], info["height"])))

    # ---- Eq. 5-6: split along the lidar tangent plane
    @torch.no_grad()
    def _split(self, params, optimizers, state, mask):
        sel = torch.where(mask)[0]
        rest = torch.where(~mask)[0]
        scales = torch.exp(params["scales"][sel])
        rotmats = normalized_quat_to_rotmat(F.normalize(params["quats"][sel], dim=-1))
        samples = torch.einsum("nij,nj,bnj->bni", rotmats, scales,
                               torch.randn(2, len(scales), 3, device=sel.device))     # default: random
        if self.tangent_split and self.lidar is not None and len(sel):
            pm = self.lidar.to_metric(params["means"][sel].detach().cpu().numpy().astype(np.float64))
            ins = self.lidar.inside(pm)
            if ins.any():
                _, k = self.lidar.tree.query(pm[ins], workers=-1)
                nrm = torch.tensor(self.lidar.normals_m[k] @ self.lidar.R.T, dtype=torch.float32, device=sel.device)
                ii = torch.tensor(np.where(ins)[0], device=sel.device)
                smax, imax = scales[ii].max(dim=-1)
                axis = rotmats[ii].gather(2, imax.view(-1, 1, 1).expand(-1, 3, 1)).squeeze(-1)  # long axis (column)
                d = axis - (axis * nrm).sum(-1, keepdim=True) * nrm                             # onto the tangent plane
                d = F.normalize(d, dim=-1)
                a = torch.randn(len(ii), device=sel.device).abs()[:, None] * smax[:, None]
                samples[0, ii] = a * d
                samples[1, ii] = -a * d

        def param_fn(name, p):
            repeats = [2] + [1] * (p.dim() - 1)
            if name == "means":
                p_split = (p[sel] + samples).reshape(-1, 3)
            elif name == "scales":
                p_split = torch.log(scales / 1.6).repeat(2, 1)
            elif name == "opacities" and self.revised_opacity:
                p_split = torch.logit(1.0 - torch.sqrt(1.0 - torch.sigmoid(p[sel]))).repeat(repeats)
            else:
                p_split = p[sel].repeat(repeats)
            return torch.nn.Parameter(torch.cat([p[rest], p_split]), requires_grad=p.requires_grad)

        def optimizer_fn(key, v):
            return torch.cat([v[rest], torch.zeros((2 * len(sel), *v.shape[1:]), device=v.device)])

        _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = torch.cat((v[rest], v[sel].repeat([2] + [1] * (v.dim() - 1))))

    @torch.no_grad()
    def _grow_gs(self, params, optimizers, state, step):
        grads = state["grad2d"] / state["count"].clamp_min(1e-12)
        is_grad_high = grads > self.grow_grad2d
        is_small = torch.exp(params["scales"]).max(dim=-1).values <= self.grow_scale3d * state["scene_scale"]
        is_dupli = is_grad_high & is_small
        is_split = is_grad_high & ~is_small
        if step < self.refine_scale2d_stop_iter:
            is_split |= state["radii"] > self.grow_scale2d
        if self.max_gs > 0:                     # GPU memory cap: keep the highest-gradient candidates that fit
            room = max(self.max_gs - len(grads), 0)
            cand = torch.where(is_dupli | is_split)[0]
            if len(cand) > room:
                keep = torch.zeros_like(is_dupli)
                keep[cand[grads[cand].argsort(descending=True)[:room]]] = True
                is_dupli &= keep
                is_split &= keep
        n_dupli, n_split = int(is_dupli.sum()), int(is_split.sum())
        if n_dupli:
            duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli)
        is_split = torch.cat([is_split, torch.zeros(n_dupli, dtype=torch.bool, device=is_split.device)])
        if n_split:
            self._split(params, optimizers, state, is_split)
        return n_dupli, n_split

    # ---- Eq. 3: remove Gaussians off the lidar surface
    @torch.no_grad()
    def _lidar_prune(self, params, optimizers, state):
        pm = self.lidar.to_metric(params["means"].detach().cpu().numpy().astype(np.float64))
        ins = np.where(self.lidar.inside(pm))[0]
        if not len(ins):
            return 0
        q = pm[ins]
        _, k = self.lidar.tree.query(q, workers=-1)
        nn = self.lidar.points_m[k]
        d_plane = np.hypot(q[:, 0] - nn[:, 0], q[:, 1] - nn[:, 1])
        d_elev = np.abs(q[:, 2] - nn[:, 2])
        lo, hi = self.lidar.column(q)
        held = (hi > q[:, 2]) & (q[:, 2] > lo - self.eps_elev)                  # D_i, under a return, above ground
        rm = ((d_plane > self.eps_plane) | (d_elev > self.eps_elev)) & ~held
        mask = torch.zeros(len(pm), dtype=torch.bool, device=params["means"].device)
        mask[torch.tensor(ins[rm], device=mask.device)] = True
        n = int(mask.sum())
        if n:
            remove(params=params, optimizers=optimizers, state=state, mask=mask)
        return n

    def step_post_backward(self, params, optimizers, state, step, info, packed=False):
        super().step_post_backward(params, optimizers, state, step, info, packed=packed)
        if (self.lidar_prune and self.lidar is not None and self.refine_start_iter < step < self.refine_stop_iter
                and step % self.lidar_every == 0):
            n = self._lidar_prune(params, optimizers, state)
            if self.verbose:
                print(f"Step {step}: {n} GSs off the lidar surface removed. Now {len(params['means'])} GSs.")
