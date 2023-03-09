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

from torch_geometric.nn import DynamicEdgeConv, global_max_pool, global_mean_pool, avg_pool_x

from utils import ktprint, set_logger, check_dirs

import torch.nn as nn
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



class Net_GCA(torch.nn.Module):
    def __init__(self, k=20, aggr='max') -> None:
        super().__init__()

        self.conv1 = DynamicEdgeConv(MLP([2 * 6, 64, 64, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 256]), k, aggr)

        reduction = 4
        self.se1 = SE(64, reduction)
        self.se2 = SE(256, reduction)

        ## pos encoding
        self.pos1 = nn.Sequential(nn.Linear(3,32),
                                    nn.BatchNorm1d(32),
                                    nn.ReLU(True),
                                    nn.Linear(32,64),
                                    nn.BatchNorm1d(64),
                                    nn.ReLU(True))

        self.pos2 = nn.Sequential(nn.Linear(3,128),
                                    nn.BatchNorm1d(128),
                                    nn.ReLU(True),
                                    nn.Linear(128,256),
                                    nn.BatchNorm1d(256),
                                    nn.ReLU(True))


        self.mlp2 = Seq(
            MLP([256, 64]),
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

        batch_num = data.y.shape[0]

        pos_b1_out_1 = self.pos1(data.x[:,0:3])
        b1_out_1 = self.conv1(data.x, data.batch)
        b1_out_1 = b1_out_1 + pos_b1_out_1
        b1_out_1 = self.se1(b1_out_1, data.batch, batch_num)

        pos_b1_out_2 = self.pos2(data.x[:,0:3])
        b1_out_2 = self.conv2(b1_out_1, data.batch)
        b1_out_2 = b1_out_2 + pos_b1_out_2
        b1_out_2 = self.se2(b1_out_2, data.batch, batch_num)




        pos_b2_out_1 = self.pos1(data.x2[:,0:3])
        b2_out_1 = self.conv1(data.x2, data.batch2)
        b2_out_1 = b2_out_1 + pos_b2_out_1
        b2_out_1 = self.se1(b2_out_1, data.batch2, batch_num)

        pos_b2_out_2 = self.pos2(data.x2[:,0:3])
        b2_out_2 = self.conv2(b2_out_1, data.batch2)
        b2_out_2 = b2_out_2 + pos_b2_out_2
        b2_out_2 = self.se2(b2_out_2, data.batch2, batch_num)

        b1_out = global_max_pool(b1_out_2, data.batch)
        b2_out = global_max_pool(b2_out_2, data.batch2)


        x_out = b2_out - b1_out

        x_out = self.mlp2(x_out)

        return F.log_softmax(x_out, dim=-1)



class SE(torch.nn.Module):
    def __init__(self, input_features, reduction):
        super().__init__()

        features = input_features // reduction


        self.conv = nn.Sequential( nn.Linear(input_features, features),
                                   nn.ReLU(True),
                                   nn.Linear(features, input_features))

        self.sigm = nn.Sigmoid()

    def forward (self, x_input, batch, batch_num):

        avg = global_meanpool(x_input, batch)

        attn = self.conv(avg)

        attn = self.sigm(attn)

        for i in range(batch_num):
            tuple = torch.where(batch[:] == i)
            x_input[tuple] = x_input[tuple] *attn[i,:]


        return x_input


def global_meanpool(x, batch):

    x_mean = global_mean_pool(x, batch)

    return x_mean



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
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--test_model', type=bool, default=False)
    parser.add_argument('--data', type=str, default='lidar')
    
    return parser.parse_args()



if __name__ == '__main__':

    train_on_real_data = True

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
        pre_transform, transform = NormalizeScale(), SamplePoints(4096)
        train_dataset = ChangeDataset(data_root_path, train=True, clearance=3, ignore_labels=ignore_labels, transform=transform, pre_transform=pre_transform)
        test_dataset = ChangeDataset(data_root_path, train=False, clearance=3, ignore_labels=ignore_labels, transform=transform, pre_transform=pre_transform)


    else:
        data_root_path = 'PCLchange/synthetic_city_scenes/'
        pre_transform, transform = NormalizeScale(), SamplePoints(4096)
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
    model = Net_GCA().to(device)

    if test_model:
        epoch = 0
        if train_on_real_data:
            modelpath = 'best_models/SiamGCN-GCA/lidar/best_gcn_model_tmp_Net_GCA.pth'
        if not train_on_real_data:
            modelpath = 'best_models/SiamGCN-GCA/synthetic/best_gcn_model_tmp_Net_GCA.pth'
        model.load_state_dict(torch.load(modelpath))
        model.eval()
        test_acc, per_cls_acc, conf = test(test_loader)
        exit()


    print(f"Using model: {model.__class__.__name__}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    scheduler = StepLR(optimizer, step_size=15, gamma=0.1)

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
