import torch
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch import nn
from torch.nn import Linear
import torch.nn as nn
from fabind.models.att_model import EfficientMCAttModel
import torch.nn.functional as F
from fabind.utils.utils import get_keepNode_tensor, gumbel_softmax_no_random
import random


class Transition_diff_out_dim(torch.nn.Module):
    # separate left/right edges (block1/block2).
    def __init__(self, embedding_channels=256, out_channels=256, n=4):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(embedding_channels)
        self.linear1 = Linear(embedding_channels, n * embedding_channels)
        self.linear2 = Linear(n * embedding_channels, out_channels)
        torch.nn.init.xavier_uniform_(self.linear1.weight, gain=0.001)
        torch.nn.init.xavier_uniform_(self.linear2.weight, gain=0.001)

    def forward(self, z):
        # z of shape b, i, j, embedding_channels, where i is protein dim, j is compound dim.
        z = self.layernorm(z)
        z = self.linear2((self.linear1(z)).relu())
        return z


class IaBNet_mean_and_pocket_prediction_cls_coords_dependent(torch.nn.Module):
    def __init__(self, args, embedding_channels=128, pocket_pred_embedding_channels=128):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(embedding_channels)
        self.args = args
        self.coordinate_scale = args.coordinate_scale
        self.normalize_coord = lambda x: x / self.coordinate_scale
        self.unnormalize_coord = lambda x: x * self.coordinate_scale
        self.stage_prob = args.stage_prob

        n_channel = 1  # ligand node has only one coordinate dimension.
        self.complex_model = EfficientMCAttModel(
            args, embedding_channels, embedding_channels, n_channel, n_edge_feats=0, n_layers=args.mean_layers,
            n_iter=args.n_iter,
            inter_cutoff=args.inter_cutoff, intra_cutoff=args.intra_cutoff, normalize_coord=self.normalize_coord,
            unnormalize_coord=self.unnormalize_coord,
        )

        self.pocket_pred_model = EfficientMCAttModel(
            args, pocket_pred_embedding_channels, pocket_pred_embedding_channels, n_channel, n_edge_feats=0,
            n_layers=args.pocket_pred_layers, n_iter=args.pocket_pred_n_iter,
            inter_cutoff=args.inter_cutoff, intra_cutoff=args.intra_cutoff, normalize_coord=self.normalize_coord,
            unnormalize_coord=self.unnormalize_coord,
        )

        self.protein_to_pocket = Transition_diff_out_dim(embedding_channels=embedding_channels, n=4, out_channels=1)

        # global nodes for protein / compound
        self.glb_c = nn.Parameter(torch.ones(1, embedding_channels))
        self.glb_p = nn.Parameter(torch.ones(1, embedding_channels))
        if args.use_esm2_feat:
            protein_hidden = 1280
        else:
            protein_hidden = 15
        if args.esm2_concat_raw:
            protein_hidden = 1295
        # self.protein_linear = nn.Linear(protein_hidden, embedding_channels) # hard-coded GVP features
        # self.compound_linear = nn.Linear(56, embedding_channels)
        self.protein_linear_whole_protein = nn.Linear(protein_hidden, embedding_channels)  # hard-coded GVP features
        self.compound_linear_whole_protein = nn.Linear(56, embedding_channels)

        self.embedding_shrink = nn.Linear(embedding_channels, pocket_pred_embedding_channels)
        self.embedding_enlarge = nn.Linear(pocket_pred_embedding_channels, embedding_channels)

        self.distmap_mlp = nn.Sequential(
            nn.Linear(embedding_channels, embedding_channels),
            nn.ReLU(),
            nn.Linear(embedding_channels, 1))

        # torch.nn.init.xavier_uniform_(self.protein_linear.weight, gain=0.001)
        # torch.nn.init.xavier_uniform_(self.compound_linear.weight, gain=0.001)
        torch.nn.init.xavier_uniform_(self.protein_linear_whole_protein.weight, gain=0.001)
        torch.nn.init.xavier_uniform_(self.compound_linear_whole_protein.weight, gain=0.001)
        torch.nn.init.xavier_uniform_(self.embedding_shrink.weight, gain=0.001)
        torch.nn.init.xavier_uniform_(self.embedding_enlarge.weight, gain=0.001)
        torch.nn.init.xavier_uniform_(self.distmap_mlp[0].weight, gain=0.001)
        torch.nn.init.xavier_uniform_(self.distmap_mlp[2].weight, gain=0.001)

    def forward(self, data, stage=1, train=False):
        keepNode_less_5 = 0
        compound_batch = data['compound'].batch
        pocket_batch = data['pocket'].batch
        complex_batch = data['complex'].batch
        protein_batch_whole = data['protein_whole'].batch
        complex_batch_whole_protein = data['complex_whole_protein'].batch

        # Pocket Prediction
        # nodes_whole = (data['protein_whole']['node_s'], data['protein_whole']['node_v'])
        # edges_whole = (data[("protein_whole", "p2p", "protein_whole")]["edge_s"], data[("protein_whole", "p2p", "protein_whole")]["edge_v"])
        # protein_out_whole = self.conv_protein(nodes_whole, data[("protein_whole", "p2p", "protein_whole")]["edge_index"], edges_whole, data.seq_whole)
        # protein_out_batched_whole, protein_out_mask_whole = to_dense_batch(protein_out_whole, protein_batch_whole)
        # pocket_cls_pred = self.protein_to_pocket(protein_out_batched_whole)
        # pocket_cls_pred = pocket_cls_pred.squeeze(-1) * protein_out_mask_whole
        # pocket_cls, _ = to_dense_batch(data.pocket_idx, protein_batch_whole)
        batched_complex_coord_whole_protein = self.normalize_coord(
            data['complex_whole_protein'].node_coords.unsqueeze(-2))
        batched_complex_coord_LAS_whole_protein = self.normalize_coord(
            data['complex_whole_protein'].node_coords_LAS.unsqueeze(-2))
        batched_compound_emb_whole_protein = self.compound_linear_whole_protein(data['compound'].node_feats)
        batched_protein_emb_whole_protein = self.protein_linear_whole_protein(data['protein_whole'].node_feats)

        # TODO self.glb_c and self.glb_p shared?
        for i in range(complex_batch_whole_protein.max() + 1):
            if i == 0:
                new_samples_whole_protein = torch.cat((
                    self.glb_c, batched_compound_emb_whole_protein[compound_batch == i],
                    self.glb_p, batched_protein_emb_whole_protein[protein_batch_whole == i]
                ), dim=0)
            else:
                new_sample_whole_protein = torch.cat((
                    self.glb_c, batched_compound_emb_whole_protein[compound_batch == i],
                    self.glb_p, batched_protein_emb_whole_protein[protein_batch_whole == i]
                ), dim=0)
                new_samples_whole_protein = torch.cat((new_samples_whole_protein, new_sample_whole_protein), dim=0)

        new_samples_whole_protein = self.embedding_shrink(new_samples_whole_protein)

        complex_coords_whole_protein, complex_out_whole_protein = self.pocket_pred_model(
            batched_complex_coord_whole_protein,
            new_samples_whole_protein,
            batch_id=complex_batch_whole_protein,
            segment_id=data['complex_whole_protein'].segment,
            mask=data['complex_whole_protein'].mask,
            is_global=data['complex_whole_protein'].is_global,
            compound_edge_index=data['complex_whole_protein', 'c2c', 'complex_whole_protein'].edge_index,
            LAS_edge_index=data['complex_whole_protein', 'LAS', 'complex_whole_protein'].edge_index,
            batched_complex_coord_LAS=batched_complex_coord_LAS_whole_protein,
            LAS_mask=None
        )

        complex_out_whole_protein = self.embedding_enlarge(complex_out_whole_protein)

        compound_flag_whole_protein = torch.logical_and(data['complex_whole_protein'].segment == 0,
                                                        ~data['complex_whole_protein'].is_global)
        compound_out_whole_protein = complex_out_whole_protein[compound_flag_whole_protein]
        protein_flag_whole_protein = torch.logical_and(data['complex_whole_protein'].segment == 1,
                                                       ~data['complex_whole_protein'].is_global)
        protein_out_whole_protein = complex_out_whole_protein[protein_flag_whole_protein]
        protein_out_batched_whole, protein_out_mask_whole = to_dense_batch(protein_out_whole_protein,
                                                                           protein_batch_whole)
        pocket_cls_pred = self.protein_to_pocket(protein_out_batched_whole)
        pocket_cls_pred = pocket_cls_pred.squeeze(-1) * protein_out_mask_whole
        pocket_cls, _ = to_dense_batch(data.pocket_idx, protein_batch_whole)

        pocket_coords_batched, _ = to_dense_batch(self.normalize_coord(data.node_xyz), pocket_batch)
        protein_coords_batched_whole, protein_coords_mask_whole = to_dense_batch(data.node_xyz_whole,
                                                                                 protein_batch_whole)

        pred_index_true = pocket_cls_pred.sigmoid().unsqueeze(-1)
        pred_index_false = 1. - pred_index_true
        pred_index_prob = torch.cat([pred_index_false, pred_index_true], dim=-1)
        # For training stability
        pred_index_prob = torch.clamp(pred_index_prob, min=1e-6, max=1 - 1e-6)
        pred_index_log_prob = torch.log(pred_index_prob)
        if self.pocket_pred_model.training:
            pred_index_one_hot = F.gumbel_softmax(pred_index_log_prob, tau=self.args.gs_tau, hard=self.args.gs_hard)
        else:
            pred_index_one_hot = gumbel_softmax_no_random(pred_index_log_prob, tau=self.args.gs_tau,
                                                          hard=self.args.gs_hard)
        pred_index_one_hot_true = (pred_index_one_hot[:, :, 1] * protein_out_mask_whole).unsqueeze(-1)
        pred_pocket_center_gumbel = pred_index_one_hot_true * protein_coords_batched_whole
        pred_pocket_center = pred_pocket_center_gumbel.sum(dim=1) / pred_index_one_hot_true.sum(dim=1)


        return pocket_cls_pred, pocket_cls, protein_out_mask_whole, protein_coords_batched_whole, pred_pocket_center

def get_model(args, logger, device):
    if args.mode == 5:
        logger.log_message("FABind")
        model = IaBNet_mean_and_pocket_prediction_cls_coords_dependent(args, args.hidden_size,
                                                                       args.pocket_pred_hidden_size)
    return model