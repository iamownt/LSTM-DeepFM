import time
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from dataset.DataSet import *
from models.LSTM import *
from models.DeepFM import *
from utils.file import *
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts


class MaskAndConcatFT(nn.Module):

    def __init__(self, tr_in, tr_out):
        super(MaskAndConcatFT, self).__init__()
        self.fc = nn.Linear(tr_in, tr_out)

    def forward(self, con_in, h_in):
        h_tr = self.fc(h_in)  # con_in(B, step*D)  h_tr(B, h_tr)
        h_tr = (h_tr - torch.mean(h_tr, dim=0))/torch.std(h_tr, dim=0)
        ori = torch.cat([con_in, h_tr], dim=1)
        return ori


class DeepPart1(nn.Module):

    def __init__(self, input_dim, hidden_dims, dropout):
        super(DeepPart1, self).__init__()
        layers = list()

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.PReLU(init=0.8))
            layers.append(nn.Dropout(p=dropout))
            input_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, input_dim)``
        """
        return self.mlp(x)


class DeepPart2(nn.Module):
    """
    It is worth noting that DeepPart2 can be just one layer of fully connected layer or multiple layers,
    depending on the complexity and performance of the data
    """

    def __init__(self, input_dim, hidden_dims, dropout, output_layer=True):
        super(DeepPart2, self).__init__()
        layers = list()
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.PReLU(init=0.8))
            layers.append(nn.Dropout(p=dropout))
            input_dim = hidden_dim
        if output_layer:
            layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, input_dim)``
        """
        return self.mlp(x)


class EncoderDecoderModelFT(nn.Module):
    """
    In fact, in order to ensure the effectiveness of the pre training weight, we usually need to finetune the added DNN
    module with a very low learning rate. On the one hand, the whole model can be fine tuned with low learning rate.
    On the other hand, we can only fine tune the added DNN module for a few epochs, and then fine tune the whole model.
    """
    def __init__(self, input_dim, hidden_dim, embedding_dim, step, hidden_tr, hidden_list, hidden_list2, dropout,
                 field_dims, embed_fm):
        super(EncoderDecoderModelFT, self).__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, embedding_dim, step)
        self.mask_concat = MaskAndConcatFT(embedding_dim, hidden_tr)
        self.deep = DeepPart1(step*input_dim + hidden_tr, hidden_list, dropout)
        # for p in self.parameters():
        #     p.requires_grad = False
        if hidden_list2:
            self.deep2 = DeepPart2(hidden_list[-1], hidden_list2, dropout)
        else:
            self.deep2 = nn.Linear(hidden_list[-1], 1)
        self.fm_linears = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        # see FeaturesEmbeddingScale for more information
        self.embedding = FeaturesEmbedding(field_dims, embed_fm)
        self.t1 = nn.Parameter(torch.tensor([0.5]))
        self.t2 = nn.Parameter(torch.tensor([0.5]))

    def forward(self, time_series, fm_part):

        batch_size = time_series.size()[0]
        c_last = self.encoder(time_series)
        time_re = time_series.reshape(batch_size, -1)
        ori = self.mask_concat(time_re, c_last)
        deep_out = self.deep(ori)
        # Below is the finetune part
        deep_dnn = self.deep2(deep_out)
        embed_fm = self.embedding(fm_part)
        deep_fm = self.fm_linears(fm_part) + self.fm(embed_fm)
        return self.t1 * deep_dnn + self.t2 * deep_fm


def SFT_Training(eval_per=False, debutanizer=False):
    if not debutanizer:
    train_x_des = "../dataset/pretrain/train_x.csv"
    train_y_des = "../dataset/pretrain/train_y.csv"
    val_x_des = "../dataset/pretrain/val_x.csv"
    val_y_des = "../dataset/pretrain/val_y.csv"
    col_leave = read_pickle_from_file("../dataset/split_features.pickle")
    input_dim = len(col_leave)
    batch_size = 256
    epochs = 40
    grad_clip = 5.
    lr = 0.01
    print_freq = 50
    best_loss = 999
    epochs_since_improvement = 0
    patience = 15
    hidden_dim = 512
    step = 4
    embedding_dim = 24
    hidden_tr = 12
    mlp_list = (256, 64)
    mlp_list2 = (32, 16)  # It is depends on the complexity of the data
    dropout = 0
    field_dims = [4, 5, 5, 2, 4, 20, 20, 20, 6, 5, 5, 10, 30]
    embed_fm = 4
    train_dst = FTDataset(train_x_des, train_y_des, col_leave, step)
    val_dst = FTDataset(val_x_des, val_y_des, col_leave, step)
    train_loader = DataLoader(train_dst, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dst, batch_size=2*batch_size, shuffle=False)
    prefix = "../checkpoint/"
    name = "sft_lstmdeepfm_" + str(step)
    ende_model = EncoderDecoderModelFT(input_dim, hidden_dim, embedding_dim, step, hidden_tr, mlp_list, mlp_list2,
                                       dropout, field_dims, embed_fm).to(device)
    if eval_per:
        eval_performance(col_leave, step, batch_size, ende_model)
        return 0
    optimizer = Adam(filter(lambda p: p.requires_grad, ende_model.parameters()), lr=lr)
    path = prefix + "lstmdeep_ed_119_0.0601.pth.tar"
    ende_model, optimizer = load_checkpoint(ende_model, path, optimizer, False)
    criterion = nn.MSELoss().to(device)
    # num_cycle = 8  # epochs/8
    # scheduler = CosineAnnealingWarmRestarts(optimizer, eta_min=1e-6, T_0=len(train_loader) * num_cycle, T_mult=2)
    history = dict(train=[], val=[], batch=[])
    for epoch in range(epochs):
        if epochs_since_improvement == patience:
            print("Reach epochs since improvement: save loss info!",)
            np.save(prefix + name + ".npy", history)
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 50 == 0:
            adjust_learning_rate(optimizer, 0.9)
        ende_model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        start = time.time()
        for i, (time_series, time_fm, label) in enumerate(train_loader):
            data_time.update(time.time() - start)
            time_series.to(device)
            output = ende_model(time_series, time_fm)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None:
                clip_gradient(optimizer, grad_clip)
            optimizer.step()
            # scheduler.step()
            losses.update(loss.item())
            batch_time.update(time.time() - start)
            start = time.time()
            # print status
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(epoch, i, len(train_loader),
                          batch_time=batch_time,
                          data_time=data_time,
                          loss=losses))
            history["batch"].append(losses.avg)
        history["train"].append(losses.avg)
        # eval the model
        ende_model.eval()
        batch_time = AverageMeter()
        val_losses = AverageMeter()
        start = time.time()
        with torch.no_grad():

            for i, (time_series, time_fm, label) in enumerate(val_loader):
                time_series.to(device)
                output = ende_model(time_series, time_fm)
                loss = criterion(output, label)
                val_losses.update(loss.item())
                batch_time.update(time.time() - start)
                start = time.time()
            val_mse = val_losses.avg
            if val_mse < best_loss:
                best_loss = val_mse
                epochs_since_improvement = 0
                torch.save({'epoch': epoch + 1, 'state_dict': ende_model.state_dict(), 'best_loss': best_loss,
                            'optimizer': optimizer.state_dict()}, os.path.join(prefix,
                                                                               str("lstmdeepfm_%d_%.4f.pth.tar" % (
                                                                                   epoch, best_loss))))
            else:
                epochs_since_improvement += 1
            print('Validation: [{0}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i,
                                                                  batch_time=batch_time,
                                                                  loss=val_losses))
        history['val'].append(val_mse)


def eval_performance(col_leave, step, batch_size, ende_model):
    path = "../checkpoint/lstmdeepfm_10_2.0578.pth.tar"
    checkpoint = torch.load(path, map_location="cpu")
    val_x_des = "../dataset/pretrain/val_x.csv"
    val_y_des = "../dataset/pretrain/val_y.csv"
    test_x_des = "../dataset/pretrain/test_x.csv"
    test_y_des = "../dataset/pretrain/test_y.csv"
    val_dst = FTDataset(val_x_des, val_y_des, col_leave, step)
    test_dst = FTDataset(test_x_des, test_y_des, col_leave, step)
    val_loader = DataLoader(val_dst, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dst, batch_size=2*batch_size, shuffle=False)
    ende_model.load_state_dict(checkpoint['state_dict'])
    ende_model.eval()
    current_rmse = 0
    current_cout = 0
    for tr1, tr2, lb in val_loader:
        train_pred = ende_model(tr1, tr2)
        current_rmse += torch.sum(torch.square(train_pred - lb))
        current_cout += len(train_pred)
    print("Val MSE: {}, VAL COUNT: {}".format(current_rmse/current_cout, current_cout))
    current_rmse = 0
    current_cout = 0
    for tr1, tr2, lb in test_loader:
        train_pred = ende_model(tr1, tr2)
        current_rmse += torch.sum(torch.square(train_pred - lb))
        current_cout += len(train_pred)
    print("Test MSE: {}, Test COUNT: {}".format(current_rmse / current_cout, current_cout))


if __name__ == "__main__":
    SFT_Training(debutanizer=True)


