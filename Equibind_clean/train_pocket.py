import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score,f1_score, accuracy_score, recall_score
import argparse
import torch.nn.functional as F
from models.equibind_new import IEGMN
from datasets.pdbbind_pocket import PDBBind
import dgl
#
# def read_strings_from_txt(path):
#     # every line will be one element of the returned list
#     with open(path) as file:
#         lines = file.readlines()
#         return [line.rstrip() for line in lines]


# class Protein(Dataset):
#     def __init__(self, data_root = '/home/tinama/project/EquiBind/data/processed/train_process',comples_root = '/home/tinama/project/EquiBind/data/timesplit_no_lig_overlap_train'):
#         # dataset parameters
#
#
#         self.root = os.listdir(data_root)
#         self.base_root=data_root
#         self.complex_names = read_strings_from_txt(comples_root)
#
#
#     def __len__(self):
#         return len(self.root)
#
#     def __getitem__(self, index):
#         file_name = os.path.join(self.base_root, 'processed_pub_mol{}.npz').format(self.complex_names[index])
#         # file_name = os.path.join(self.base_root,'processed_pub_mol{}.npz'.format(name))
#         data = np.load(file_name)
#
#         coords = data['coords']
#         x_feat = data['x_feat']
#         label = data['label']
#
#         coords = torch.from_numpy(coords)
#         x_feat = torch.from_numpy(x_feat)
#         label = torch.from_numpy(label)
#
#         return coords, x_feat, label

def site_loss(preds, labels):

    loss = 0
    preds_concates = []
    labels_concates = []
    idx = 0

    for label in labels:

        pred = preds[idx: idx + len(label)]
        idx += len(label)
        label = label.cuda().float()

        pos_preds = pred[label == 1]
        pos_labels = label[label == 1]
        neg_preds = pred[label == 0]
        neg_labels = label[label == 0]

        n_points_sample = len(pos_labels)
        pos_indices = torch.randperm(len(pos_labels))[:n_points_sample]
        neg_indices = torch.randperm(len(neg_labels))[:n_points_sample]

        pos_preds = pos_preds[pos_indices]
        pos_labels = pos_labels[pos_indices]
        neg_preds = neg_preds[neg_indices]
        neg_labels = neg_labels[neg_indices]

        preds_concat = torch.cat([pos_preds, neg_preds])
        labels_concat = torch.cat([pos_labels, neg_labels])
        labels_concat = labels_concat.reshape(-1,1)

        loss += F.binary_cross_entropy_with_logits(preds_concat, labels_concat)
        preds_concates.append(preds_concat)
        labels_concates.append(labels_concat)

    return loss/len(labels), preds_concates, labels_concates




class Model(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
        self.iegmn = IEGMN(n_lays=8, debug=False, device='cuda:0', use_rec_atoms=False, shared_layers=False, noise_decay_rate=0.5, cross_msgs=True, noise_initial=1,
                 use_edge_features_in_gmn=True, use_mean_node_features=True, residue_emb_dim=64, iegmn_lay_hid_dim=64, num_att_heads=30,
                 dropout=0.1, nonlin='lkyrelu', leakyrelu_neg_slope=0.01, **kwargs)

        self.mlp = nn.Sequential(
                    nn.Linear(64, 128),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(128, 128),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(128, 64),
                )

        self.net_out = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 16),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(16, 1),
        )

    def forward(self, lig_graph, rec_graph, geometry_graph=None, complex_names=None,epoch=0):
        x1 = self.iegmn(lig_graph, rec_graph, geometry_graph, complex_names, epoch)
        # x2 = self.mlp(x1)
        x3 = self.net_out(x1)
        return x3


def iterate(
    net,
    dataset,
    optimizer,
    args,
    test=False,
    save_path=None,
    pdb_ids=None,
    summary_writer=None,
    epoch_number=None,
):
    """Goes through one epoch of the dataset, returns information for Tensorboard."""

    if test:
        net.eval()
        torch.set_grad_enabled(False)
    else:
        net.train()
        torch.set_grad_enabled(True)

    # Statistics and fancy graphs to summarize the epoch:
    info = []
    total_processed_pairs = 0
    # Loop over one epoch:
    for it, (lig_graph, rec_graph, geometry_graph, complex_names,label) in enumerate(tqdm(dataset)):  # , desc="Test " if test else "Train")):

        # try:

            # lig_graph = lig_graph.to('cuda:0')
            # x_feat = x_feat.to('cuda:0')
            # label = label.to('cuda:0')



            if not test:
                optimizer.zero_grad()

            outputs = net(lig_graph, rec_graph, geometry_graph, complex_names, epoch_number)
            # label = label.float()

            loss, sampled_preds, sampled_labels = site_loss(outputs, label)
            idx = 0
            accuracy_list = []
            f1_list = []
            recall_list = []

            for l in label:
                output = outputs[idx: idx + len(l)]
                idx += len(l)
                predicted_labels = (output > 0).float()
                a=l.reshape(-1,1).float()
                # if predicted_labels.sum()>0:
                accuracy_1 = accuracy_score(a.detach().cpu().view(-1).numpy(),
                                           predicted_labels.detach().cpu().view(-1).numpy())
                f1_1 = f1_score(a.detach().cpu().view(-1).numpy(), predicted_labels.detach().cpu().view(-1).numpy())
                recall_1 =recall_score(a.detach().cpu().view(-1).numpy(), predicted_labels.detach().cpu().view(-1).numpy())
                accuracy_list.append(accuracy_1)
                f1_list.append(f1_1)
                recall_list.append(recall_1)
                # else:
                #     accuracy_1 = 0.0
                #     f1_1 = 0.0


            # accuracy=np.array(accuracy_list).sum()/len(label)
            # f1=np.array(f1_list).sum()/len(label)

            accuracy = np.mean(accuracy_list)
            f1 = np.mean(f1_list)
            recall = np.mean(recall_list)



            # Compute the gradient, update the model weights:
            if not test:
                loss.backward()
                do_step = True
                for param in net.parameters():
                    if param.grad is not None:
                        if (1 - torch.isfinite(param.grad).long()).sum() > 0:
                            do_step = False
                            break
                if do_step is True:
                    optimizer.step()
                # loss.backward()
                # optimizer.step()

            try:
                if sampled_labels is not None:
                    roc_auc = []
                    f1_s = []
                    acc_s = []
                    recall_s =[]

                    for index, item in enumerate(sampled_labels):
                        #if item.shape[0] == 0:
                            #continue
                        true = np.rint(item.detach().cpu().view(-1).numpy())
                        pred = sampled_preds[index].detach().cpu().view(-1).numpy()
                        pred_s = (pred>0).astype(np.float32)
                        roc_auc.append(
                            roc_auc_score(true,pred)
                        )

                        acc_s.append(accuracy_score(true,pred_s))
                        f1_s.append(f1_score(true,pred_s))
                        recall_s.append(recall_score(true,pred_s))


                    roc_auc = np.mean(roc_auc)
                    accuracy_sample = np.mean(acc_s)
                    f1_sample = np.mean(f1_s)
                    recall_sample = np.mean(recall_s)


                else:
                    roc_auc = 0.0
            except Exception as e:
                print("Problem with computing roc-auc")
                print(e)
                continue

            info.append(
                dict(
                    {
                        "Loss": loss.item(),
                        "ROC-AUC": roc_auc,
                        "ACC":accuracy,
                        "F1_score":f1,
                        "Recall":recall,
                        "ACC_sample":accuracy_sample,
                        "F1_score_sample":f1_sample,
                        "Recall_sample":recall_sample
                    }
                )
            )
        # except Exception as e:
        #     print("Problem with cuda")
        #     print(e)
        #     continue
     # Turn a list of dicts into a dict of lists:
    newdict = {}
    for k, v in [(key, d[key]) for d in info for key in d]:
        if k not in newdict:
            newdict[k] = [v]
        else:
            newdict[k].append(v)
    info = newdict

    # Final post-processing:
    return info

def graph_collate_revised(batch):
    lig_graphs, rec_graphs, geometry_graph, complex_names, label = map(list, zip(*batch))
    geometry_graph = dgl.batch(geometry_graph) if geometry_graph[0] != None else None
    return dgl.batch(lig_graphs), dgl.batch(rec_graphs),geometry_graph, complex_names, label







if __name__ == "__main__":

    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int,default=42)
    p.add_argument('--experiment_name', type=str, help='name that will be added to the runs folder output')
    args = p.parse_args()
    # args.add_argument('--logdir', type=str, default='runs', help='tensorboard logdirectory')
    writer = SummaryWriter("runs/{}".format(args.experiment_name))
    model_path = "trainpocket_model/" + args.experiment_name
    if not Path("trainpocket_model/").exists():
        Path("trainpocket_model/").mkdir(exist_ok=False)

    # torch.backends.cudnn.deterministic = True
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    # np.random.seed(args.seed)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    # dgl.random.seed(seed)
    # random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    net = Model()
    net = net.to('cuda:0')

    # net.load_state_dict(torch.load('/home/tinama/project/simsiam_4.4/simsiam/models/Simsiam_site_dmasif_pre_1024_layer3_256_epoch352.pth')['model_state_dict'])

    trainset = PDBBind(process_dir = '/home/tinama/project/EquiBind/data/processed/sizeNone_INDEXtimesplit_no_lig_overlap_train_Hpolar0_H1_BSPprot0_BSPlig0_surface0_pocketRad4_ligRad5_recRad30_recMax10_ligMaxNone_chain10_POCKETmatch_atoms_to_lig')
    valset = PDBBind(process_dir='/home/tinama/project/EquiBind/data/processed/sizeNone_INDEXtimesplit_no_lig_overlap_val_Hpolar0_H1_BSPprot0_BSPlig0_surface0_pocketRad4_ligRad5_recRad30_recMax10_ligMaxNone_chain10_POCKETmatch_atoms_to_lig')
    testset = PDBBind(process_dir='/home/tinama/project/EquiBind/data/processed/test')
    # trainset, tempset= train_test_split(dataset, test_size=0.2, random_state=42)
    # valset, testset = train_test_split(tempset, test_size=0.5, random_state=42)
    train_loader = DataLoader(trainset, batch_size=32, shuffle=True, collate_fn=graph_collate_revised,
                              pin_memory=True, num_workers=0)
    val_loader = DataLoader(valset, batch_size=32, collate_fn=graph_collate_revised,
                            pin_memory=True, num_workers=0)
    test_loader = DataLoader(testset, batch_size=32, collate_fn=graph_collate_revised,
                            pin_memory=True, num_workers=0)

    optimizer = torch.optim.Adam(net.parameters(), lr=2e-4, amsgrad=True)
    #schedule = MultiStepLR(optimizer, milestones=[25], gamma=0.01)
    best_loss = 1e10
    starting_epoch = 0

    for i in range(starting_epoch, 300):
        #schedule.step()
        # Train first, Test second:
        for dataset_type in ["Train", "Validation", "Test"]:
            if dataset_type == "Train":
                test = False
            else:
                test = True

            suffix = dataset_type
            if dataset_type == "Train":
                dataloader = train_loader
            elif dataset_type == "Validation":
                dataloader = val_loader
            elif dataset_type == "Test":
                dataloader = test_loader

            # Perform one pass through the data:
            info = iterate(
                net,
                dataloader,
                optimizer,
                args,
                test=test,
                summary_writer=writer,
                epoch_number=i,
            )

            # Write down the results using a TensorBoard writer:
            for key, val in info.items():
                if key in [
                    "Loss",
                    "ROC-AUC",
                    "ACC",
                    "F1_score",
                    "Recall",
                    "ACC_sample",
                    "F1_score_sample",
                    "Recall_sample"
                ]:
                    writer.add_scalar(f"{key}/{suffix}", np.mean(val), i)

                if "R_values/" in key:
                    val = np.array(val)
                    writer.add_scalar(f"{key}/{suffix}", np.mean(val[val > 0]), i)

            if dataset_type == "Validation":  # Store validation loss for saving the model
                val_loss = np.mean(info["Loss"])

        if True:  # Additional saves
            if val_loss < best_loss:
                print("Validation loss {}, saving model".format(val_loss))
                torch.save(
                    {
                        "epoch": i,
                        "model_state_dict": net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_loss": best_loss,
                    },
                    model_path + "_epoch{}.pth".format(i),
                )
                # best_loss = val_loss






