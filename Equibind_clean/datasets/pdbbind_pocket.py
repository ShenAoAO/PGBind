import os
import random
from copy import deepcopy, copy

from dgl import save_graphs, load_graphs

from joblib import Parallel, delayed, cpu_count
import torch
import dgl
from biopandas.pdb import PandasPdb
from joblib.externals.loky import get_reusable_executor

from rdkit import Chem
from rdkit.Chem import MolFromPDBFile
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F

from commons.geometry_utils import random_rotation_translation, rigid_transform_Kabsch_3D_torch
from commons.process_mols import get_rdkit_coords, get_receptor, get_pocket_coords, \
    read_molecule, get_rec_graph, get_lig_graph_revised, get_receptor_atom_subgraph, get_lig_structure_graph, \
    get_geometry_graph, get_lig_graph_multiple_conformer, get_geometry_graph_ring
from commons.utils import pmap_multi, read_strings_from_txt, log


class PDBBind(Dataset):
    """"""

    def __init__(self, process_dir='/home/tinama/project/EquiBind/data/processed/sizeNone_INDEXtimesplit_no_lig_overlap_val_Hpolar0_H1_BSPprot0_BSPlig0_surface0_pocketRad4_ligRad5_recRad30_recMax10_ligMaxNone_chain10_POCKETmatch_atoms_to_lig'):

        self.processed_dir = process_dir
        print(f'using processed directory: {self.processed_dir}')
        self.lig_graph_path = 'lig_graphs_rdkit_coords.pt'
        log('loading data into memory')
        coords_dict = torch.load(os.path.join(self.processed_dir, 'pocket_and_rec_coords.pt'))
        self.pockets_coords = coords_dict['pockets_coords']
        self.lig_graphs, _ = load_graphs(os.path.join(self.processed_dir, self.lig_graph_path))
        self.rec_graphs, _ = load_graphs(os.path.join(self.processed_dir, 'rec_graphs.pt'))
        self.geometry_graphs, _ =  load_graphs(os.path.join(self.processed_dir, 'geometry_regularization.pt'))
        self.geometry_graphs, _ =  load_graphs(os.path.join(self.processed_dir, 'geometry_regularization_ring.pt'))
        self.complex_names = coords_dict['complex_names']
        assert len(self.lig_graphs) == len(self.rec_graphs)
        log('finish loading data into memory')
        self.cache = {}


    def __len__(self):
        return len(self.lig_graphs)

    def __getitem__(self, idx):
        pocket_coords = self.pockets_coords[idx]
        lig_graph = deepcopy(self.lig_graphs[idx])
        rec_graph = self.rec_graphs[idx]
        label =rec_graph.ndata['label']
        # Randomly rotate and translate the ligand.
        rot_T, rot_b = random_rotation_translation(translation_distance=5.0)
        lig_coords_to_move =lig_graph.ndata['new_x']
        mean_to_remove = lig_coords_to_move.mean(dim=0, keepdims=True)
        lig_graph.ndata['new_x'] = (rot_T @ (lig_coords_to_move - mean_to_remove).T).T + rot_b
        new_pocket_coords = (rot_T @ (pocket_coords - mean_to_remove).T).T + rot_b
        geometry_graph = self.geometry_graphs[idx]
        return lig_graph.to('cuda:0'), rec_graph.to('cuda:0'), geometry_graph, self.complex_names[idx], label




