import torch
import torch.nn.functional as F


class SkeletonBased(object):

    def _parse_kintree_table(self, kintree_table, rm_bones):
        assert len(kintree_table[0]) == len(kintree_table[1])
        num_joints = len(kintree_table[0])

        parents = kintree_table[0]

        children = [[] for _ in range(num_joints)]
        Jconnect = torch.eye(num_joints).byte()
        for i in range(num_joints):
            p =  kintree_table[0][i]
            c =  kintree_table[1][i]
            if p < 0:
                continue
            children[p].append(c)
            Jconnect[p, c] = 1
            Jconnect[c, p] = 1
        
        # Bconnect = Jconnect.clone()
        # for i in sorted(rm_bones, reverse=True):
        #     Bconnect = torch.cat([Bconnect[:i, :], Bconnect[i+1:, :]], dim=0)
        #     Bconnect = torch.cat([Bconnect[:, :i], Bconnect[:, i+1:]], dim=1)
        
        # store the mapping from the old index of a bone to the new one
        Bid_table = list(range(num_joints))
        for i in sorted(rm_bones, reverse=False):
            Bid_table[i] = None
            Bid_table[i+1:] = [j-1 for j in Bid_table[i+1:]]

        return parents, children, Jconnect, Bid_table
    
    def _batch_transform_bones(self, Jtr_rest, Jrot_rest, Jtr, Jrot, Jpose, parents, children, Jconnect, rm_bones, Bid_table):
        ''' Construct a local coordinate (Btr, Brot) and conditions (Bcond) for each bone.
        
            args:
                Jtr: torch.tensor (batch, joints, 3)
                Jrot: torch.tensor (batch, joints, 3, 3)
                parents: torch.tensor (joints)
                children: [[int, ...], ...]
        '''
        Btr_list = []
        Brot_list = []
        Bneigh_list = []
        Bcond_list = []
        Blim_list = []
        Bsec_list = []

        bone_idx = [i for i in range(Jtr.shape[1]) if i not in rm_bones]  # # remove hands and foot
        for i in bone_idx:
            # === bone coordinate (posed) ===
            Btr_i, Brot_i = self._get_bone_coordinate(i, Jtr, Jrot, parents, children)
            Brot_i_inv = Brot_i.permute(0, 1, 3, 2)
            Jtr_local = (Brot_i_inv @ (Jtr - Btr_i)[..., None])[..., 0]  # (B, J, 3)
            Jrot_local = Brot_i_inv @ Jrot  # (B, J, 3, 3)

            # === bone coordinate (rest) ===
            Btr_i_rest, Brot_i_rest = self._get_bone_coordinate(i, Jtr_rest, Jrot_rest, parents, children)
            Brot_i_inv_rest = Brot_i_rest.permute(0, 1, 3, 2)
            Jtr_local_rest = (Brot_i_inv_rest @ (Jtr_rest - Btr_i_rest)[..., None])[..., 0]  # (B, J, 3)
            Jrot_local_rest = Brot_i_inv_rest @ Jrot_rest  # (B, J, 3, 3)

            # # === bone properties ===
            Bneigh_i = self._get_bone_neighbor_transfm(i, Jtr_local, Jrot_local, Jpose, parents, children, rm_bones, Bid_table)
            Bcond_i = self._get_bone_condition(Jtr, Jrot, Btr_i, Brot_i_inv, bone_idx)
            Blim_i = self._get_bone_limit(i, Jtr_local, parents, children, rm_bones)
            Bsec_i = self._get_bone_section_normal(i, Jtr_local_rest, Jrot_local_rest, parents, children, Jconnect, rm_bones)
            # Bsec_i = self._get_bone_lim_normal(i, Jtr_local, Jrot_local, parents, children, Jconnect, rm_bones)

            # === ===
            Btr_list.append(Btr_i)
            Brot_list.append(Brot_i)
            Bneigh_list.append(Bneigh_i)
            Bcond_list.append(Bcond_i)
            Blim_list.append(Blim_i)
            Bsec_list.append(Bsec_i)

        Btr = torch.cat(Btr_list, dim=1)
        Brot = torch.cat(Brot_list, dim=1)
        Bneigh = self.cat_variable_size(Bneigh_list, dim=1, pad_dim=2)
        Bcond = torch.cat(Bcond_list, dim=1)
        # Blim = torch.cat(Blim_list, dim=1)
        Blim = self.cat_variable_size(Blim_list, dim=1, pad_dim=2)
        # Bsec = torch.cat(Bsec_list, dim=1)
        Bsec = self.cat_variable_size(Bsec_list, dim=1, pad_dim=2)

        return Btr, Brot, Bneigh, Bcond, Blim, Bsec
    
    def _get_bone_coordinate(self, i, Jtr, Jrot, parents, children):
        """ Get the local coordinate of a bone.

                J_pi ---> J_i ===> J_ci
        """

        pi = parents[i]  # parent of the i-th joint (a digit)
        ci = children[i]  # children of the i-th joint (a list)

        if pi == -1:  # no parent joint (joint #0)
            # Btr_i = Jtr[:, ci].mean(dim=1, keepdim=True)
            Btr_i = Jtr[:, [i]]
        else:  # has parent joint
            if len(ci) == 0:  # no children
                # Btr_i = Jtr[:, [i]]
                Btr_i = (Jtr[:, [i]] + self._get_child_Jtr(i, Jtr, Jrot, parents, children)) / 2
            else:  # has children
                Btr_i = Jtr[:, ci+[i]].mean(dim=1, keepdim=True)

        Brot_i = Jrot[:, [i]]
        return Btr_i, Brot_i
    
    def _get_bone_neighbor_transfm(self, i, Jtr, Jrot, Jpose, parents, children, rm_bones, Bid_table):
        """ Get Jtr and Jpose of the i-th Bone's neighbor bones.
        
            Jtr_pi ---> Jtr_i (Jpose_i) ===> Jtr_ci (Jpose_ci) ---> Jtr_cci

        """
        pi = parents[i]
        ci = children[i]

        Jneigh_list = []

        # 1. the parent side
        if pi != -1:  # has parent (except joint #0)
            Jtr_this = self._get_child_Jtr(i, Jtr, Jrot, parents, children)
            Jtr_center = Jtr[:, [i]]
            Jtr_other = Jtr[:, [pi]]
            Jpose_local = -Jpose[:, [i]]  # rot i->pi is the inverse of rot pi->i
            # id of this and the other bone
            Bid = torch.tensor([Bid_table[i], Bid_table[pi], torch.nan]).reshape(1, 1, 3).expand_as(Jtr_center)
            
            Jneigh_list.append(torch.stack([Jtr_this, Jtr_center, Jtr_other, Jpose_local, Bid], dim=-2))  # (B, 1, 5, D)
        
        # 2. the child side
        Jtr_this = Jtr[:, [i]]

        for j in ci:
            if j in rm_bones:
                continue
            Jtr_center = Jtr[:, [j]]
            Jtr_other = self._get_child_Jtr(j, Jtr, Jrot, parents, children)
            Jpose_local = Jpose[:, [j]]  # rot i->ci
            # id of this and the other bone
            Bid = torch.tensor([Bid_table[i], Bid_table[j], torch.nan]).reshape(1, 1, 3).expand_as(Jtr_center)

            Jneigh_list.append(torch.stack([Jtr_this, Jtr_center, Jtr_other, Jpose_local, Bid], dim=-2))  # (B, 1, 5, D)

        # collect
        if len(Jneigh_list) == 0:
            Jneigh_list.append(torch.zeros(Jtr[:, 0, None, None].shape).repeat(1, 1, 5, 1) * torch.nan)
        Jneigh = torch.cat(Jneigh_list, dim=1).unsqueeze(1)  # (B, 1, ?, 5, D)
        return Jneigh
    
    def _get_child_Jtr(self, i, Jtr, Jrot, parents, children):
        """ Get the child Jtr of joint i. Handle corners cases including multiple or no children.
        """
        pi = parents[i]
        ci = children[i]

        if len(ci) == 0:  # no children (joint #10,11,15,22,23)
            # copy the direction of the last bone and rotate it with the end point
            Jtr_child = (Jrot[:, [i]] @ Jrot[:, [pi]].transpose(-1, -2) @ (Jtr[:, [i]] - Jtr[:, [pi]]).unsqueeze(-1)).squeeze(-1) + Jtr[:, [i]]  # cause artifacts
            # Jtr_child = (Jrot[:, [i]] @ (Jtr[:, [i]] - Jtr[:, [pi]]).unsqueeze(-1)).squeeze(-1) + Jtr[:, [i]]  # no artifacts (but may not be correct)
            # Jtr_child = (Jtr[:, [i]] - Jtr[:, [pi]]) + Jtr[:, [i]]  # straight end
        elif len(ci) > 1:  # multiple children (joint #9)
            Jtr_child = Jtr[:, ci].mean(1, keepdim=True)
        else:  # only one child
            Jtr_child = Jtr[:, ci]
        return Jtr_child
    
    def _get_bone_condition(self, Jtr, Jrot, Btr_i, Brot_i_inv, bone_idx):
        # -1- adjacent joints ---
        # Jrot_cond = torch.cat([Jrot_pi, Jrot_i], dim=1)
        # Jtr_cond = torch.cat([Jtr_pi, Jtr_i], dim=1)

        # -2- all joints ---
        Jrot_cond = Jrot[:, bone_idx]
        Jtr_cond = Jtr[:, bone_idx]

        # transform to local coordinate
        Jrot_cond_local = Brot_i_inv @ Jrot_cond  # (B, J, 3, 3)
        # Jquat_cond_local = matrix_to_quaternion(Jrot_cond_local)  # (B, J, 4)
        # JaxisAng_cond_local = quaternion_to_axis_angle(Jquat_cond_local)  # (B, J, 4)
        # Jeuler_cond_local = matrix_to_euler_angles(Jrot_cond_local, convention='XYZ')  # (B, J, 3)
        Jtr_cond_local = (Brot_i_inv @ (Jtr_cond - Btr_i)[..., None])[..., 0]  # (B, J, 3)

        B = Jtr.shape[0]
        Bcond_i = torch.cat([
            Jrot_cond_local.reshape(B, 1, -1),
            # Jquat_cond_local.reshape(B, 1, -1),
            # JaxisAng_cond_local.reshape(B, 1, -1),
            # Jeuler_cond_local.reshape(B, 1, -1),
            Jtr_cond_local.reshape(B, 1, -1),
            ], 
        dim=-1)
        return Bcond_i
    
    def _get_bone_limit(self, i, Jtr, parents, children, rm_bones):
        """ Get the limit points of a bone.

                J_pi ---> J_i ===> J_ci
        """
        pi = parents[i]  # parent of the i-th joint (a digit)
        ci = children[i]  # children of the i-th joint (a list)

        limit_list = []
        if pi != -1:  # has parent
            limit_list.append(Jtr[:, [i]])
        if len(ci) > 0:
            for j in ci:
                if j not in rm_bones:
                    limit_list.append(Jtr[:, [j]])
        Blim_i = torch.cat(limit_list, dim=-2)  # (B, points, D)
        
        Blim_i = Blim_i.unsqueeze(1)  # (B, bones, points, D)
        return Blim_i
    
    def _get_bone_section_normal(self, i, Jtr, Jrot, parents, children, Jconnect, rm_bones):
        """ Get the section normal vectors of the i-th bone's two joints.
        """
        pi = parents[i]  # a digit
        ci = children[i]  # a list

        normal_list = []
        if pi != -1:  # has parent
            if len(ci) > 1:
                assert i == 9  # joint #9 has three children
                assert 12 in ci  # we use the middle one, #12
                normal_list.append(self._joints_to_direction(i, 12, Jtr, Jrot, Jconnect, children))  # (B, 1, 3)  # TODO: extend to support besides SMPL
            elif len(ci) == 1:
                normal_list.append(self._joints_to_direction(i, ci[0], Jtr, Jrot, Jconnect, children))  # (B, 1, 3)

        if len(ci) > 0:
            for j in ci:
                if j not in rm_bones:
                    normal_list.append(self._joints_to_direction(j, i, Jtr, Jrot, Jconnect, children))  # (B, 1, 3)
        else:
            # trick: borrow the normal from the parent bone
            normal_list.append(-self._joints_to_direction(i, pi, Jtr, Jrot, Jconnect, children))  # (B, 1, 3)
            
        Bsec_i = torch.cat(normal_list, dim=-2).unsqueeze(1) # (B, 1, ?, 3)
        return Bsec_i
    
    def _joints_to_direction(self, J_center, J_this, Jtr, Jrot, Jconnect, children, weighted_by_length=True):
        """ Get the section normal across J_center.

                J_other ---> J_center <=== J_this
        """
        J_other = Jconnect[J_center].nonzero().flatten().tolist()
        J_other.remove(J_center)
        J_other.remove(J_this)

        J_this = [J_this]
        # if J_this has many children, then these children should also be included
        if len(children[J_this[0]]) > 1:
            for j in children[J_this[0]]:
                if j not in J_this:
                    J_this.append(j)
        Jtr_this = Jtr[:, J_this].mean(1, keepdim=True)
        
        vec_this = Jtr[:, [J_center]] - Jtr_this  # (B, 1, 3)
        norm_this = vec_this.norm(p=2, dim=-1, keepdim=True)
        assert (norm_this != 0).all()
        dir_this = vec_this / norm_this

        if len(J_other) == 0:  # J_center is an end joint, cannot get vec_other from Jtr
            # copy the direction of the last bone and rotate it with the end point
            dir_end = (Jrot[:, [J_center]] @ Jrot[:, J_this].transpose(-1, -2) @ dir_this[..., None])[..., 0]

            vec_section = dir_this + dir_end
        else:
            # Choice A:
            Jtr_other = Jtr[:, J_other]

            # Choice B: 
            # if a joint in J_other has many children, then these children should also be included
            # for i in J_other:
            #     if len(children[i]) > 1:
            #         for j in children[i]:
            #             if j not in J_other:
            #                 J_other.append(j)
            # Jtr_other = Jtr[:, J_other].mean(1, keepdim=True)

            vec_other = -(Jtr[:, [J_center]] - Jtr_other)  # (B, 1, 3), inverse direction
            vec_all = torch.cat([vec_this, vec_other], dim=-2)
            norm_all = vec_all.norm(p=2, dim=-1, keepdim=True)
            assert (norm_all != 0).all()
            dir_all = vec_all / norm_all

            if weighted_by_length:
                vec_section = vec_all.mean(dim=1, keepdim=True)
            else:
                vec_section = dir_all.mean(dim=1, keepdim=True)

        # normalize
        norm_section = vec_section.norm(p=2, dim=-1, keepdim=True)
        assert (norm_section != 0).all()
        dir_section = vec_section / norm_section

        return dir_section

    def cat_variable_size(self, tensor_list, dim=1, pad_dim=2):
        max_size = max([x.shape[pad_dim] for x in tensor_list])
        
        ndim = [x.ndim for x in tensor_list]
        assert min(ndim) == max(ndim), f'the tensors have inconsistent ndim: {ndim}'
        ndim = ndim[0]

        if pad_dim >= 0:
            pad_dim = pad_dim - ndim 

        for i, x in enumerate(tensor_list):
            this_size = x.shape[pad_dim]
            pad = [0, 0] * (abs(pad_dim)-1) + [0, max_size-this_size]
            tensor_list[i] = F.pad(x, pad, value=torch.nan)

        return torch.cat(tensor_list, dim=dim)
            