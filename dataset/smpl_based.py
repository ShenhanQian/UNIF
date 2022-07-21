from scipy.spatial import KDTree

from .surface3d import Surface3D
from .skeleton_based import SkeletonBased


class SmplBased(Surface3D, SkeletonBased):

    def _get_smpl_partId(self, smpl_weights, parents, Bid_table):

        def map_fn(i):
            if Bid_table[i] is None:
                return map_fn(parents[i])
            else:
                return Bid_table[i]

        Bid_map = [map_fn(i) for i in range(len(Bid_table))]
        
        _, smpl_partId = smpl_weights.max(1)
        smpl_partId.apply_(lambda x: Bid_map[x])

        return smpl_partId

    def _runtime_preprocess(self, item):
        item = self._spatial_sampling(item)

        if self.label_for_train:
            item = self._get_label(item)
        
        if self.partId_from_smpl:
            item = self._get_partId(item)
        
        item = self._clean_item(item)
        return item
    
    def _get_partId(self, item):
        pts_smpl_kdtree = KDTree(item['pts_smpl'])
        dist, smpl_vert_ids = pts_smpl_kdtree.query(item['pts_surface'])
        item['partId'] = item['partId_smpl'][smpl_vert_ids]
        return item