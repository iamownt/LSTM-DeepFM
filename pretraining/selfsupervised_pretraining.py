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
from dataset.DataSet import PretrainDataset
from models.LSTM import *
from utils.file import *


class MaskAndConcat(nn.Module):
    """
    :param mask_c: time_series mask rate
    :param mask_h: c_last mask rate
    :param tr_in, tr_out: the c_last's simple linear transfer
    :param tup: step * input_dim
    """
    def __init__(self, mask_c, mask_h, tr_in, tr_out, tup):
        super(MaskAndConcat, self).__init__()
        self.mask_c = mask_c
        self.mask_h = mask_h
        self.tup = tup
        self.prod = self.tup[0] * self.tup[1]
        self.fc = nn.Linear(tr_in, tr_out)

    def forward(self, con_in, h_in):
        task1 = torch.rand(1) >= 0.5
        if task1:
            h_tr = self.fc(h_in)  # con_in(B, step*D)  h_tr(B, h_tr)
            h_tr = (h_tr - torch.mean(h_tr, dim=0))/torch.std(h_tr, dim=0)
            mask1 = (torch.rand(*con_in.size()) >= self.mask_c).to(device)
            mask2 = (torch.rand(*h_tr.size()) >= self.mask_h).to(device)
            masked = torch.cat([con_in*mask1, h_tr*mask2], dim=1)
            ori = torch.cat([con_in, h_tr], dim=1)
        else:
            B = h_in.shape[0]
            h_tr = self.fc(h_in)  # con_in(B, step*D)  h_tr(B, h_tr)
            h_tr = (h_tr - torch.mean(h_tr, dim=0))/torch.std(h_tr, dim=0)
            mask1 = torch.ones(B, *self.tup).bool()
            mask1[torch.arange(B), torch.randint(0, self.tup[0], (B,))] = 0
            mask1 = mask1.reshape(B, -1).to(device)
            mask2 = (torch.rand(*h_tr.size()) >= self.mask_h).to(device)
            masked = torch.cat([con_in*mask1, h_tr*mask2], dim=1)
            ori = torch.cat([con_in, h_tr], dim=1)
        return ori, masked, mask1, mask2


class DeepComponent(nn.Module):
    """
    This class is composed of DeepComponentEncoder and DeepComponentDecoder.
    """
    def __init__(self, input_dim, hidden_dims, dropout):
        super(DeepComponent, self).__init__()
        layers = list()
        input_ori = input_dim
        for hidden_dim in (hidden_dims + hidden_dims[::-1][1:]):
            layers.append(nn.Linear(input_dim, hidden_dim))
            # layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(0.8))
            layers.append(nn.Dropout(p=dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, input_ori))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, input_dim)``
        """
        return self.mlp(x)


class SSPLoss(nn.Module):
    """
    The self supervised pretraining loss.
    """
    def __init__(self, df_std, h_dim, step, bernoulli=True):
        super(SSPLoss, self).__init__()
        self.bernoulli = bernoulli
        self.std_1 = torch.from_numpy(df_std.values).float()
        self.std_1 = torch.cat([self.std_1]*step).to(device)
        self.std_2 = torch.ones(h_dim).to(device)

    def forward(self, flag, preds, labels, mask):

        y1 = preds * ~mask
        y2 = labels * ~mask
        count = torch.sum(~mask)
        error = y1 - y2
        return torch.sum((error/(self.std_1+1e-9))**2)/count if flag else torch.sum((error/(self.std_2+1e-9))**2)/count


class EncoderDecoderModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, embedding_dim, step, mask_c, mask_h, hidden_tr, hidden_list, dropout):
        super(EncoderDecoderModel, self).__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, embedding_dim, step)
        for p in self.parameters():
            p.requires_grad = True
        self.mask_concat = MaskAndConcat(mask_c, mask_h, embedding_dim, hidden_tr, (step, input_dim))
        self.deep = DeepComponent(step*input_dim + hidden_tr, hidden_list, dropout)

    def forward(self, time_series):
        batch_size = time_series.size()[0]
        c_last = self.encoder(time_series)
        time_re = time_series.reshape(batch_size, -1)
        ori, con_inp, mask1, mask2 = self.mask_concat(time_re, c_last)
        deep_out = self.deep(con_inp)

        return deep_out, ori, mask1, mask2


def SSP_Training(debutanizer=False):
    train_des = "../dataset/Debutanizer_Data/train_x.csv"
    val_des = "../dataset/Debutanizer_Data/val_x.csv"
    if not debutanizer:
        col_leave = read_pickle_from_file("../dataset/split_features.pickle")
        input_dim = len(col_leave)
        batch_size = 256
        epochs = 300
        grad_clip = 5.
        lr = 3e-4
        print_freq = 50
        best_loss = 999
        epochs_since_improvement = 0
        patience = 15
        hidden_dim = 512
        step = 4
        step_size = 10
        embedding_dim = 24
        hidden_tr = 12
        mask_c = 0.15
        mask_h = 0.15
        mlp_list = (256, 64)
        dropout = 0
    else:
        col_leave = None
        input_dim = 13
        batch_size = 128
        epochs = 300
        grad_clip = 5.
        lr = 0.01
        print_freq = 30
        best_loss = 999
        epochs_since_improvement = 0
        patience = 15
        hidden_dim = 13
        embedding_dim = 12
        step = 2
        step_size = 1
        hidden_tr = 12
        mask_c = 0.15
        mask_h = 0.15
        mlp_list = (7, 5)
        dropout = 0

    train_dst = PretrainDataset(train_des, step, col_leave, step_size)
    val_dst = PretrainDataset(val_des, step, col_leave, step_size)
    train_loader = DataLoader(train_dst, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dst, batch_size=2*batch_size, shuffle=False)
    prefix = "../checkpoint/"
    name = "ssp_lstmdeep_" + str(mask_c) + "_" + str(step)
    ende_model = EncoderDecoderModel(input_dim, hidden_dim, embedding_dim, step, mask_c, mask_h, hidden_tr, mlp_list, dropout).to(device)
    optimizer = Adam(filter(lambda p: p.requires_grad, ende_model.parameters()), lr=lr)
    path = prefix + "lstm_ed_9_0.0365.pth.tar"
    ende_model, optimizer = load_checkpoint(ende_model, path, optimizer, False)
    criterion = SSPLoss(train_dst.std*10, hidden_tr, step).to(device)
    save_dic = dict(train=[], val=[], batch=[])
    for epoch in range(epochs):
        if epochs_since_improvement == patience:
            print("Reach epochs since improvement: save loss info!",)
            np.save(prefix + name+".npy", save_dic)
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 5 == 0:
            adjust_learning_rate(optimizer, 0.9)
        ende_model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        start = time.time()
        for i, time_series in enumerate(train_loader):
            data_time.update(time.time() - start)
            time_series.to(device)
            re_con, ori_con, mask1, mask2 = ende_model(time_series)
            loss_tr1 = criterion(1, re_con[:, :-hidden_tr], ori_con[:, :-hidden_tr], mask1)
            loss_tr2 = criterion(0, re_con[:, -hidden_tr:], ori_con[:, -hidden_tr:], mask2)
            loss = (loss_tr1 + loss_tr2)
            print(loss_tr1, loss_tr2)
            if loss_tr1.isnan() or loss_tr2.isnan():
                print("aa")
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None:
                clip_gradient(optimizer, grad_clip)
            optimizer.step()
            losses.update(loss.item())
            batch_time.update(time.time() - start)
            start = time.time()
            # print status
            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      .format(epoch, i, len(train_loader),
                              batch_time=batch_time,
                              data_time=data_time,
                              loss=losses))
            save_dic['batch'].append(losses.avg)
        save_dic['train'].append(losses.avg)
        # eval the model
        ende_model.eval()
        batch_time = AverageMeter()
        val_losses = AverageMeter()
        start = time.time()
        with torch.no_grad():

            for i, time_series in enumerate(val_loader):
                time_series.to(device)
                re_con, ori_con, mask1, mask2 = ende_model(time_series)
                loss_va1 = criterion(1, re_con[:, :-hidden_tr], ori_con[:, :-hidden_tr], mask1)
                loss_va2 = criterion(0, re_con[:, -hidden_tr:], ori_con[:, -hidden_tr:], mask2)
                loss = loss_va1 + loss_va2
                val_losses.update(loss.item())
                batch_time.update(time.time() - start)
                start = time.time()
            val_mse = val_losses.avg
            if val_mse < best_loss:
                best_loss = val_mse
                epochs_since_improvement = 0
                torch.save({'epoch': epoch+1, 'state_dict': ende_model.state_dict(), 'best_loss': best_loss,
                            'optimizer': optimizer.state_dict()}, os.path.join(prefix,
                                                                               str("lstmdeep_ed_%d_%.4f.pth.tar" %
                                                                                   (epoch, best_loss))))
            else:
                epochs_since_improvement += 1
            print('Validation: [{0}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i,
                                                                  batch_time=batch_time,
                                                                  loss=val_losses))
        save_dic['val'].append(val_losses.avg)


if __name__ == "__main__":
    SSP_Training(debutanizer=True)


