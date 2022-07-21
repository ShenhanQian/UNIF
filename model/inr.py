import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import axis_angle_to_matrix

from model.group_linear import GroupLinear
from utils.embedder import get_embedder


class INR(nn.Module):
    def __init__(
        self,
        d_in,
        dims,
        num_parts,
        skip_in=(),
        d_cond=4,
        multires=0,
        geometric_init=True,
        radius_init=1,
        beta=100,
        weight_norm=False,
        num_groups=1,
    ):
        super().__init__()

        dims = [d_in] + dims + [1]
        self.d_in = d_in
        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.d_cond = d_cond
        self.num_groups = num_groups

        # -------- fourier feature --------
        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            dims[0] = input_ch
        
            if num_groups > 1:
                self.ff_projector = GroupLinear(input_ch-d_in, input_ch-d_in, num_group=num_groups)
            else:
                self.ff_projector = nn.Linear(input_ch-d_in, input_ch-d_in)
            torch.nn.init.normal_(self.ff_projector.weight, mean=0, std=1e-5)
            torch.nn.init.constant_(self.ff_projector.bias, 0.)

        # -------- condition --------
        if d_cond > 0:
            if d_in == 2:
                d_cond_in = num_parts*(4+2)
            elif d_in == 3:
                d_cond_in = num_parts*(9+3)  # rot, tr
                # d_cond_in = num_parts*(4+3)  # quaternion, tr
                # d_cond_in = num_parts*(3+3)  # euler/axisAngle, tr
                # d_cond_in = num_parts*(3)  # tr
            else:
                raise NotImplementedError

            dims[0] = dims[0] + self.d_cond

            if num_groups > 1:
                self.cond_projector = nn.Sequential(
                    GroupLinear(d_cond_in, self.d_cond*4, num_group=num_groups),
                    nn.Softplus(beta=beta),
                    GroupLinear(self.d_cond*4, self.d_cond*4, num_group=num_groups),
                    nn.Softplus(beta=beta),
                    GroupLinear(self.d_cond*4, self.d_cond, num_group=num_groups),
                )
            else:
                self.cond_projector = nn.Sequential(
                    nn.Linear(d_cond_in, self.d_cond*4),
                    nn.Softplus(beta=beta),
                    nn.Linear(self.d_cond*4, self.d_cond*4),
                    nn.Softplus(beta=beta),
                    nn.Linear(self.d_cond*4, self.d_cond),
                )

            torch.nn.init.normal_(self.cond_projector[-1].weight, mean=0, std=1e-5)
            torch.nn.init.constant_(self.cond_projector[-1].bias, 0.)

        # -------- layers --------
        for layer in range(0, self.num_layers - 1):

            if layer + 1 in skip_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
            
            if num_groups > 1:
                lin = GroupLinear(dims[layer], out_dim, num_group=num_groups)
            else:
                lin = nn.Linear(dims[layer], out_dim)

            if geometric_init:
                if layer == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
                    torch.nn.init.constant_(lin.bias, -radius_init)
                else:
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.bias, 0.0)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(layer), lin)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()

    def forward(self, input, cond, **kwargs):
        # -------- Fourier feature --------
        if self.embed_fn is not None:
            input = self.embed_fn(input)
            input = torch.cat([
                input[..., :self.d_in],
                self.ff_projector(input[..., self.d_in:]),
            ], dim=-1)

        # -------- Bcond --------
        if self.d_cond > 0:
            cond = self.cond_projector(cond)
            input = torch.cat([input, cond], dim=-1)

        # -------- layers --------
        x = input
        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))

            if layer in self.skip_in:
                x = torch.cat([x, input], -1) / np.sqrt(2)

            x = lin(x)
            if layer < self.num_layers - 2:
                x = self.activation(x)

        return x


class UNIF(nn.Module):
    def __init__(
        self,
        num_parts,
        d_in,
        dims,
        skip_in=(),
        d_cond=4,
        multires=0,
        geometric_init=True,
        radius_init=1,
        beta=100,
        weight_norm=False,
        blend_alpha=2,
    ):
        super().__init__()
        self.num_parts = num_parts

        # -------- list of MLPs --------
        # self.unif = nn.ModuleList([
        #     INR(d_in, dims, num_parts, skip_in, d_cond, multires, geometric_init, radius_init, beta, weight_norm) for _ in range(num_parts)
        # ])

        # -------- GroupLinear MLP (faster) --------
        self.unif = INR(d_in, dims, num_parts, skip_in, d_cond, multires, geometric_init, radius_init, beta, weight_norm, num_groups=num_parts)
        # ----------------

        self.rigid_alpha = nn.Parameter(torch.ones(num_parts, num_parts) * blend_alpha)
        self.rigid_beta = nn.Parameter(torch.zeros(num_parts, num_parts))

    def forward(self, pts, is_testing=True, return_parts=False, **priors):
        # from global to local
        assert self.num_parts <= priors['Btr'].shape[1]
        Bcond = priors['Bcond'][:, :self.num_parts]
        Btr = priors['Btr'][:, :self.num_parts]
        Brot = priors['Brot'][:, :self.num_parts]
        Bneigh = priors['Bneigh'][:, :self.num_parts]  # (B, J, num_groups, 4, D), the second to last dim: [Jtr_this, Jtr_center, Jtr_other, Jpose_local]

        Btr = Btr.unsqueeze(2)    # (B, J, 1, D)
        Brot_inv = Brot.unsqueeze(2).transpose(3, 4)  # (B, J, 1, D, D)
        pts = pts.unsqueeze(1)  # (B, 1, V, D)

        pts_local = torch.matmul(Brot_inv, (pts - Btr).unsqueeze(-1)).squeeze(-1)  # (B, J, V, D)
        
        # Adjacent Part Seaming (APS)
        deform_offset, mask_valid, weights = self._get_deform_offset(pts_local, Bneigh)
        pts_local = pts_local + deform_offset

        # -------- point query --------
        B, J, V, D = pts_local.shape
        Bcond = Bcond.unsqueeze(2).repeat(1, 1, V, 1)
        
        if isinstance(self.unif, nn.ModuleList):
            # list of MLPs
            pred_local_parts = []
            for i in range(pts_local.shape[1]):
                pred_local = self.unif[i](pts_local[:, i], Bcond[:, i])  # (B, V, 1)

                pred_local_parts.append(pred_local.reshape(B, 1, V, 1))
            pred_parts = torch.cat(pred_local_parts, dim=1)  # (B, J, V, 1)
        else:
            # MlP with GroupLinear
            pts_local_ = pts_local.permute(1, 0, 2, 3).reshape(J, B*V, D)
            Bcond_ = Bcond.permute(1, 0, 2, 3).reshape(J, B*V, -1)
            pred_parts = self.unif(pts_local_, Bcond_).reshape(J, B, V, 1).permute(1, 0, 2, 3)  # (B, J, V, 1)
        
        
        # -------- fusion --------
        # hard blending
        # pred, part_indices = pred_parts.min(dim=1)

        # soft blending
        # pred, part_indices = pred_parts.min(dim=1)
        # w = nn.Softmin(1)(pred_parts * 50)
        # pred = (pred_parts * w).sum(1)

        # soft delta blending
        pred_min, part_indices = pred_parts.min(dim=1)
        delta_pred = pred_parts - pred_min[:, None]
        # delta_pred = pred_parts - pred_min[:, None].detach()
        w = nn.Softmin(1)(delta_pred * 200)
        pred = (delta_pred * w).sum(1) + pred_min

        # use label for partId
        # if is_testing:
        #     pred, part_indices = pred_parts.min(dim=1)
        # else:
        #     part_indices = priors['partId'].unsqueeze(-1)
        #     pred = pred_parts.gather(1, part_indices.unsqueeze(1)).squeeze(1)

        if return_parts:
            return pred, part_indices, pred_parts, pts_local
        else:
            return pred
    
    def forward_limit(self, pts_limit, **priors):
        assert self.num_parts <= priors['Btr'].shape[1]
        pts_local = pts_limit[:, :self.num_parts]  # (B, num_parts, num_points, D)
        B, J, V, D = pts_local.shape

        Bcond = priors['Bcond'][:, :self.num_parts]  # (B, J, 1, D)
        Bcond = Bcond.unsqueeze(2).repeat(1, 1, V, 1)  # (B, J, V, D)

        if isinstance(self.unif, nn.ModuleList):
            # list of MLPs
            pred_lim = []
            for i in range(self.num_parts):
                pred_local_i = self.unif[i](pts_local[:, i], Bcond[:, i]).unsqueeze(1)  # (B, 1, V, 1)
                pred_lim.append(pred_local_i)
            pred_lim = torch.cat(pred_lim, dim=1)  # (B, J, V, 1)
        else:            
            # GroupLinear MlP
            pts_local_ = pts_local.permute(1, 0, 2, 3).reshape(J, B*V, D)
            Bcond_ = Bcond.permute(1, 0, 2, 3).reshape(J, B*V, -1)
            pred_lim = self.unif(pts_local_, Bcond_).reshape(J, B, V, 1).permute(1, 0, 2, 3)  # (B, J, V, 1)
        return pred_lim
    
    def _get_deform_offset(self, pts, Bneigh):
        """
            Jtr_pi ---> Jtr_i (Jpose_i) ===> Jtr_ci (Jpose_ci) ---> Jtr_cci
        
        args:
            pts:    (B, num_parts, num_points, D)
            Bneigh: (B, num_parts, num_groups, 4, D), the second to last dim: [Jtr_this, Jtr_center, Jtr_other, Jpose_local]
        """
        num_groups = Bneigh.shape[2]
        offset_groups = []
        w_other_groups = []
        mask_valid = None
        for gi in range(num_groups):
            Jtr_this, Jtr_center, Jtr_other, Jpose_local, Bid = Bneigh[:, :, gi].split(1, dim=-2)
            
            w_other = self._get_blending_weights(pts, Jtr_center, Jtr_other, Jtr_this, Bid)  # weight of the adjacent bone (0 - 1)
            mask_valid = w_other.nan_to_num() <= 0.5 if mask_valid is None else mask_valid * (w_other.nan_to_num() <= 0.5)
            w_other_groups.append(w_other)

            Jpose_local_w = Jpose_local * w_other
            Jrot_back = axis_angle_to_matrix(-Jpose_local_w.nan_to_num())  # warp back to the rest pose
                                                            # wo/ nan_to_num(), the returned rot matrix
                                                            # will be all zero
            offset_gi = (Jrot_back @ (pts - Jtr_center).unsqueeze(-1)).squeeze(-1) + Jtr_center - pts
            offset_groups.append(offset_gi)  # (B, num_parts, num_points, D)
        
        offsets = torch.stack(offset_groups, dim=2)  # (B, num_parts, num_groups, num_points, D)
        offsets = offsets.nan_to_num()
        offset_sum = offsets.sum(dim=2)
        w_others = torch.stack(w_other_groups, dim=2)  # (B, num_parts, num_groups, num_points, 1)
        return offset_sum, mask_valid, w_others
    
    def _get_blending_weights(self, pts, Jtr_center, Jtr_neigh, Jtr_self, Bid, eps=1e-8):
        """ 
        Compute the blending weights based on the relative position of a point 
        to a bones and a neighbour bone.

            Jtr_neighbor <=== Jtr_center ---> Jtr_self
        """
        # get par id
        Bid_this, Bid_other, _ = Bid[:, :, 0].split(1, dim=-1)  # (B, num_parts, 1)

        Bid_this = Bid_this[0, :, 0]
        Bid_other = Bid_other[0, :, 0]
        Bid_this = Bid_this.nan_to_num().long()
        Bid_other = Bid_other.nan_to_num().long()

        # rigidness coefficients (alpha)
        rigid_alpha = F.elu(self.rigid_alpha - 1) + 1  # ensure positive
        rigid_alpha_n = rigid_alpha[Bid_other, Bid_this].reshape(1, -1, 1, 1)
        rigid_alpha_s = rigid_alpha[Bid_this, Bid_other].reshape(1, -1, 1, 1)

        # rigidness coefficients (beta)
        rigid_beta_n = self.rigid_beta[Bid_other, Bid_this].reshape(1, -1, 1, 1)
        rigid_beta_s = self.rigid_beta[Bid_this, Bid_other].reshape(1, -1, 1, 1)

        # compute weights
        vec_bone_n = Jtr_neigh - Jtr_center
        vec_bone_s = Jtr_self - Jtr_center
        norm_bone_n = vec_bone_n.norm(p=2, dim=-1, keepdim=True)
        norm_bone_s = vec_bone_s.norm(p=2, dim=-1, keepdim=True)

        vec_basis_n = (Jtr_neigh - Jtr_self) / (norm_bone_n + norm_bone_s + eps) * norm_bone_n
        vec_basis_s = -vec_basis_n / (norm_bone_n + eps) * norm_bone_s
        norm_basis_n = vec_basis_n.norm(p=2, dim=-1, keepdim=True)
        norm_basis_s = vec_basis_s.norm(p=2, dim=-1, keepdim=True)

        tr_o = Jtr_neigh - vec_basis_n
        vec_pts = pts.detach() - tr_o

        dir_basis_n = vec_basis_n / (norm_basis_n + eps)
        dir_basis_s = vec_basis_s / (norm_basis_s + eps)

        ratio_neigh = (vec_pts * dir_basis_n).sum(dim=-1, keepdim=True) / (norm_basis_n + eps)
        ratio_self = (vec_pts * dir_basis_s).sum(dim=-1, keepdim=True) / (norm_basis_s + eps)

        mask_invalid = ratio_neigh.isnan() * ratio_self.isnan()
        rigid_neigh = torch.exp(ratio_neigh.nan_to_num() * rigid_alpha_n + rigid_beta_n)
        rigid_self = torch.exp(ratio_self.nan_to_num() * rigid_alpha_s + rigid_beta_s)

        w = rigid_neigh / (rigid_neigh + rigid_self)
        w[mask_invalid] = 0.
        return w


if __name__ == '__main__':
    from config.pipeline.default import get_cfg_defaults
    from model import get_model
    
    cfg = get_cfg_defaults()
    cfg.merge_from_file('config/model/inr.py')
    cfg.freeze()
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)

    model = get_model(cfg.MODEL.name)(**cfg.MODEL.kwargs).cuda()

    x = torch.rand([100, 3]).cuda()
    x = model(x)
