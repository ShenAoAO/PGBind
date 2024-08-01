
import torch
from datasets.samplers import HardSampler
from trainer.trainer import Trainer
import copy

class BindingTrainer(Trainer):
    def __init__(self, **kwargs):
        super(BindingTrainer, self).__init__(**kwargs)

    def forward_pass(self, batch):
        pocket_label,lig_graphs, rec_graphs, ligs_coords, recs_coords, ligs_pocket_coords, recs_pocket_coords, geometry_graphs, complex_names = tuple(
            batch)
        # lig_graphs_orignal = copy.deepcopy(lig_graphs)
        # rec_graphs_orignal = copy.deepcopy(rec_graphs)
        # geometry_graphs_orignal = copy.deepcopy(geometry_graphs)
        # pocket_pred = self.pocket_model(lig_graphs, rec_graphs, geometry_graphs,complex_names=complex_names,epoch=self.epoch)
        # a = pocket_pred.reshape(-1, 1).float()
        # idx = 0
        # pocket_label = []
        # for rec in recs_coords:
        #     output = a[idx: idx + rec.size()[0]]
        #     pred = (output > 0).float()
        #     idx += len(rec)
        #     pocket_label.append(pred)
        #
        #
        # ### get number of pockets
        # def cal_cluster_num(affinity, binary=False):
        #     if binary:
        #         affinity = affinity > 0.5
        #     x = affinity + 1e-7  #
        #     D = torch.diag(x.sum(-1) ** -0.5)
        #     L_norm = torch.eye(x.shape[0]).cuda() - torch.matmul(torch.matmul(D, x), D)
        #     u, s, v = torch.svd(L_norm)
        #     max_cha = 0
        #     num_cluster = 0
        #     for i in range(0, x.shape[0] - 1):
        #         if s[-(2 + i)] - s[-(1 + i)] > max_cha:
        #             max_cha = s[-(2 + i)] - s[-(1 + i)]
        #             num_cluster = i
        #     return num_cluster + 1
        # #
        # def distance(X, Y):
        #     dis = (X ** 2).sum(-1, keepdim = True) + (Y ** 2).sum(-1, keepdim = True).T \
        #           - 2 * torch.matmul(X, Y.T)
        #     return (dis + 0.01) ** 0.5
        #
        # for i, pocket_ in enumerate(pocket_label):
        #     pocket_coord = recs_coords[i][pocket_.squeeze() > 0.5]
        #     distance_matrix = distance(pocket_coord, pocket_coord)
        #     distance_matrix_threshold = distance_matrix < distance_matrix.max() * 0.5
        #     num_pocket = cal_cluster_num(distance_matrix_threshold)
        #
        #     from sklearn.cluster import SpectralClustering
        #     import open3d as o3d
        #     clustering = SpectralClustering(n_clusters=num_pocket, assign_labels="discretize",
        #                         affinity='precomputed').fit(distance_matrix_threshold.cpu().numpy())
        #     for j in range(max(clustering.labels_) + 1):
        #         pcd0 = o3d.geometry.PointCloud()
        #         idx = clustering.labels_ == j
        #         pcd0.points = o3d.utility.Vector3dVector(pocket_coord.cpu().numpy()[idx] + 0.3)
        #         pcd0.paint_uniform_color([0,0,1])
        #
        #         pcd1  = o3d.geometry.PointCloud()
        #         pcd1.points = o3d.utility.Vector3dVector(recs_coords[i].cpu().numpy())
        #         pcd1.paint_uniform_color([1,0,0])
        #
        #         o3d.visualization.draw_geometries([pcd0, pcd1])
        #
        #


        ligs_coords_pred, ligs_keypts, recs_keypts, rotations, translations, geom_reg_loss= self.model(pocket_label, lig_graphs, rec_graphs, geometry_graphs,
                                                                                             complex_names=complex_names,
                                                                                             epoch=self.epoch)
        loss, loss_components = self.loss_func(ligs_coords, recs_coords, ligs_coords_pred, ligs_pocket_coords,
                                               recs_pocket_coords, ligs_keypts, recs_keypts, rotations, translations, geom_reg_loss,
                                               self.device)
        return loss, loss_components, ligs_coords_pred, ligs_coords

    def after_batch(self, ligs_coords_pred, ligs_coords, batch_indices):
        cutoff = 5
        centroid_distances = []
        for lig_coords_pred, lig_coords in zip(ligs_coords_pred, ligs_coords):
            centroid_distances.append(torch.linalg.norm(lig_coords_pred.mean(dim=0) - lig_coords.mean(dim=0)))
        centroid_distances = torch.tensor(centroid_distances)
        above_cutoff = torch.tensor(batch_indices)[torch.where(centroid_distances > cutoff)[0]]
        if isinstance(self.sampler, HardSampler):
            self.sampler.add_hard_indices(above_cutoff.tolist())

    def after_epoch(self):
        if isinstance(self.sampler, HardSampler):
            self.sampler.set_hard_indices()
