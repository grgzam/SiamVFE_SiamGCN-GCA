import os
import os.path as osp

from datetime import datetime
import argparse
import pickle

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Dropout
from torch.optim.lr_scheduler import StepLR

import torch_geometric

from change_dataset import ChangeDataset, MyDataLoader, ChangeDataset_synthetic
from transforms import NormalizeScale, SamplePoints
from metric import ConfusionMatrix
from imbalanced_sampler import ImbalancedDatasetSampler
from pointnet2 import MLP

from torch_geometric.nn import DynamicEdgeConv, global_max_pool

from utils import ktprint, set_logger, check_dirs

import torch.nn as nn

import torch_scatter
import numpy as np




#### log file setting
print  = ktprint
cur_filename = osp.splitext(osp.basename(__file__))[0]
log_dir = 'logs'
check_dirs(log_dir)
log_filename = osp.join(log_dir, '{}_{date:%Y-%m-%d_%H_%M_%S}'.format(cur_filename, date=datetime.now())+'.logging')
set_logger(log_filename)

#### log file setting finished!
#     0           1       2         3         4
# ["nochange","removed","added","change","color_change"]

NUM_CLASS = 5
USING_IMBALANCE_SAMPLING = True


class PFNLayerV2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()

        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.relu = nn.ReLU()

    def forward(self, inputs, unq_inv):

        x = self.linear(inputs)
        x = self.norm(x) if self.use_norm else x
        x = self.relu(x)
        x_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]

        if self.last_vfe:
            return x_max
        else:
            x_concatenated = torch.cat([x, x_max[unq_inv, :]], dim=1)
            return x_concatenated


class Net_PILLAR(torch.nn.Module):
    def __init__(self, k=20, aggr='max') -> None:
        super().__init__()
        #                  -x   -y    x    y
        self.range_xy = [-3.0, -3.0, 3.0, 3.0]

        self.voxel_x = 6.0
        self.voxel_y = 6.0

        self.x_offset = self.range_xy[0]
        self.y_offset = self.range_xy[1]
        self.z_offset = 0.0

        self.nx = int((self.range_xy[2] - self.range_xy[0])/self.voxel_x)
        self.ny = int((self.range_xy[3] - self.range_xy[1])/self.voxel_y)

        self.scale_xy = int(self.nx * self.ny)
        self.scale_y = self.ny

        self.nz = 1

        num_filters = [12, 64, 64]
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayerV2(in_filters, out_filters, use_norm = True, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)


        ## For vox size 6.0
        self.conv_2d = nn.Sequential(   nn.Conv2d(64, 1024, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(1024, eps=1e-3, momentum=0.01),
                                        nn.ReLU())

        self.mlp2 = Seq(
            MLP([1024, 64]),
            Lin(64, NUM_CLASS))

    def forward(self, data):
        """
        Args:
            data: [x], BN x 6, point clouds of 2016
                  [x2], BN x 6, point clouds of 2020
                  [batch], BN1, batch index of point clouds in 2016
                  [batch2], BN2, batch index of point clouds in 2020
                  [y], B, label
        Returns:
            out: []， Bx[NUM_CLASS]
        """
        # A simplified network

        batch_number = data.y.shape[0]

        b1_features, b1_unq_inv, b1_voxel_coords = self.obtain_dyn_pillar_indices (data.x[:,0:3], data.x[:,3:6], data.batch)
        for pfn in self.pfn_layers:
            b1_features = pfn(b1_features, b1_unq_inv)


        b1_bev = self.pillar_scatter_to_bev(b1_features, b1_voxel_coords)
        b1_bev_conv = self.conv_2d(b1_bev)

        b1_bev_conv = b1_bev_conv.squeeze(3)
        b1_bev_conv = b1_bev_conv.squeeze(2)



        b2_features, b2_unq_inv, b2_voxel_coords = self.obtain_dyn_pillar_indices (data.x2[:,0:3], data.x2[:,3:6], data.batch2)
        for pfn in self.pfn_layers:
            b2_features = pfn(b2_features, b2_unq_inv)


        b2_bev = self.pillar_scatter_to_bev(b2_features, b2_voxel_coords)

        b2_bev_conv = self.conv_2d(b2_bev)

        b2_bev_conv = b2_bev_conv.squeeze(3)
        b2_bev_conv = b2_bev_conv.squeeze(2)


        x_out = b2_bev_conv - b1_bev_conv

        x_out = self.mlp2(x_out)

        return F.log_softmax(x_out, dim=-1)



    def pillar_scatter_to_bev(self, features_in, coords_in):

        pillar_features, coords = features_in, coords_in

        batch_spatial_features = []
        batch_size = coords_in[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                features_in.shape[1],
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, features_in.shape[1] * self.nz, self.ny, self.nx)

        return batch_spatial_features


    def obtain_dyn_pillar_indices (self, points_input, features_input, batch):

        points = points_input.clone()

        point_cloud_range = torch.tensor([self.x_offset, self.y_offset]).to(points.device)
        voxel_size = torch.tensor([self.voxel_x,self.voxel_y]).to(points.device)
        grid_size = torch.tensor([self.nx, self.ny]).to(points.device)

        points = torch.cat((batch.unsqueeze(1), points), dim=1)

        points_coords = torch.floor((points[:, [1,2]] - point_cloud_range[[0,1]]) / voxel_size[[0,1]]).int()

        mask = ((points_coords >= 0) & (points_coords < grid_size[[0,1]])).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        features_mask = features_input[mask]

        points_xyz = points[:, [1, 2, 3]].contiguous()

        merge_coords = points[:, 0].int() * self.scale_xy + \
                       points_coords[:, 0] * self.scale_y + \
                       points_coords[:, 1]

        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

        points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)

        f_cluster = points_xyz - points_mean[unq_inv, :]

        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        f_center[:, 2] = points_xyz[:, 2] - self.z_offset


        features = [points[:, 1:], f_cluster, f_center, features_mask]

        features = torch.cat(features, dim=-1)

        # generate voxel coordinates
        unq_coords = unq_coords.int()

        voxel_coords = torch.stack((torch.div(unq_coords, self.scale_xy, rounding_mode='trunc'),
                                   torch.div((unq_coords % self.scale_xy), self.scale_xy, rounding_mode='trunc'),
                                   unq_coords % self.scale_y,
                                   torch.zeros(unq_coords.shape[0]).to(unq_coords.device).int()
                                   ), dim=1)

        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]


        return features, unq_inv, voxel_coords



def train(epoch, loader):
    model.train()

    confusion_matrix = ConfusionMatrix(NUM_CLASS + 1)

    correct = 0
    for i,data in enumerate(loader):

        data = data.to(device)
        optimizer.zero_grad()        

        out = model(data)
        loss = F.nll_loss(out, data.y)
        pred = out.max(1)[1]
        correct += pred.eq(data.y).sum().item()        

        loss.backward()
        optimizer.step()

        confusion_matrix.increment_from_list(data.y.cpu().detach().numpy() + 1, pred.cpu().detach().numpy() + 1)

    train_acc = correct / len(loader.dataset)
    print('Epoch: {:03d}, Train: {:.4f}, per_class_acc: {}'.format(epoch, train_acc, confusion_matrix.get_per_class_accuracy()))



def test(loader):
    model.eval()
    confusion_matrix = ConfusionMatrix(NUM_CLASS+1)
    correct = 0
    for test_i, data in enumerate(loader):

        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()

        confusion_matrix.increment_from_list(data.y.cpu().detach().numpy() + 1, pred.cpu().detach().numpy() + 1)
    test_acc = correct / len(loader.dataset)

    print('Epoch: {:03d}, Test: {:.4f}, per_class_acc: {}'.format(epoch, test_acc, confusion_matrix.get_per_class_accuracy()))

    return test_acc, confusion_matrix.get_per_class_accuracy(), confusion_matrix



def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--test_model', type=bool, default=False)
    parser.add_argument('--data', type=str, default='lidar')
    
    return parser.parse_args()



if __name__ == '__main__':

    ignore_labels = []
    my_args = get_args()
    print(my_args)

    test_model = my_args.test_model

    if my_args.data == 'lidar':
        train_on_real_data = True
    elif my_args.data == 'synthetic':
        train_on_real_data = False
    else:
        print ("Invalid --data argument, exiting...")
        exit()



    if train_on_real_data:
        data_root_path = 'PCLchange/lidar/'
        pre_transform, transform = NormalizeScale(), SamplePoints(16384)
        train_dataset = ChangeDataset(data_root_path, train=True, clearance=3, ignore_labels=ignore_labels, transform=transform, pre_transform=pre_transform)
        test_dataset = ChangeDataset(data_root_path, train=False, clearance=3, ignore_labels=ignore_labels, transform=transform, pre_transform=pre_transform)


    else:
        data_root_path = 'PCLchange/synthetic_city_scenes/'
        pre_transform, transform = NormalizeScale(), SamplePoints(16384)
        train_dataset = ChangeDataset_synthetic(data_root_path, train=True, clearance=3, ignore_labels=ignore_labels, transform=transform, pre_transform=pre_transform)
        test_dataset = ChangeDataset_synthetic(data_root_path, train=False, clearance=3, ignore_labels=ignore_labels, transform=transform, pre_transform=pre_transform)


    a = len(test_dataset)

    NUM_CLASS = len(train_dataset.class_labels)

    train_sampler = ImbalancedDatasetSampler(train_dataset)
    test_sampler = ImbalancedDatasetSampler(test_dataset)
    test_sampler.set_sampler_like(train_sampler)


    if not USING_IMBALANCE_SAMPLING:
        train_loader = MyDataLoader(train_dataset, batch_size=my_args.batch_size, shuffle=True, num_workers=my_args.num_workers)
        test_loader = MyDataLoader(test_dataset, batch_size=my_args.batch_size, shuffle=True, num_workers=my_args.num_workers)
    else:
        train_loader = MyDataLoader(train_dataset, batch_size=my_args.batch_size, shuffle=False, num_workers=my_args.num_workers, sampler=train_sampler)
        test_loader = MyDataLoader(test_dataset, batch_size=my_args.batch_size, shuffle=False, num_workers=my_args.num_workers, drop_last=True)

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net_PILLAR().to(device)

    if test_model:
        epoch = 0
        if train_on_real_data:
            modelpath = 'best_models/SiamVFE/lidar/best_gcn_model_tmp_Net_PILLAR.pth'
        if not train_on_real_data:
            modelpath = 'best_models/SiamVFE/synthetic/best_gcn_model_tmp_Net_PILLAR.pth'
        model.load_state_dict(torch.load(modelpath))
        model.eval()
        test_acc, per_cls_acc, conf = test(test_loader)
        exit()

    print(f"Using model: {model.__class__.__name__}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    scheduler = StepLR(optimizer, step_size=15, gamma=0.1)


    # test_accs = []
    max_acc = 0
    max_per_cls = None
    epoch_best = 1
    for epoch in range(1, 601):
        print ("epoch is:", epoch)
        train(epoch, train_loader) # Train one epoch
        test_acc, per_cls_acc, conf = test(test_loader)

        scheduler.step() # Update learning rate

        if test_acc > max_acc:
            torch.save(model.state_dict(), f'best_gcn_model_tmp_{model.__class__.__name__}.pth')

            with open(f'best_gcn_model_conf_tmp_{model.__class__.__name__}.pickle', 'wb') as f:
                pickle.dump(conf, f, protocol=pickle.HIGHEST_PROTOCOL)

            max_acc = test_acc
            max_per_cls = per_cls_acc
            epoch_best = epoch

    print('Epoch: {:03d}, get best acc: {:.4f}, per class acc: {}'.format(epoch_best, max_acc, max_per_cls))
