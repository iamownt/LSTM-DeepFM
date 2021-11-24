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


def USP_Training(debutanizer=False):
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
        embedding_dim = 24
        step = 4
        step_size = 10
    else:
        col_leave = None
        input_dim = 13
        batch_size = 32
        epochs = 300
        grad_clip = 5.
        lr = 0.02
        print_freq = 30
        best_loss = 999
        epochs_since_improvement = 0
        patience = 15
        hidden_dim = 13
        embedding_dim = 12
        step = 2
        step_size = 1

    prefix = "../checkpoint/"
    os.makedirs(prefix, exist_ok=True)
    name = "usp_lstm_" + str(embedding_dim) + "_" + str(step)
    lstm_ed = LSTMEncoderDecoder(input_dim, hidden_dim, embedding_dim, step).to(device)
    optimizer = Adam(lstm_ed.parameters(), lr=lr)
    criterion = nn.MSELoss().to(device)
    train_dst = PretrainDataset(train_des, step, col_leave, step_size)
    val_dst = PretrainDataset(val_des, step, col_leave, step_size)
    print("Train Dataset Size:{}, Val Dataset Size:{}".format(len(train_dst), len(val_dst)))
    train_loader = DataLoader(train_dst, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dst, batch_size=2*batch_size, shuffle=False)
    save_dic = dict(train=[], val=[], batch=[])
    for epoch in range(epochs):
        if epochs_since_improvement == patience:
            print("Reach epochs since improvement: save loss info!", )
            np.save(prefix + name + ".npy", save_dic)
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 5 == 0:
            adjust_learning_rate(optimizer, 0.9)
        lstm_ed.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        start = time.time()
        for i, time_series in enumerate(train_loader):
            data_time.update(time.time() - start)
            time_series.to(device)
            re_con = lstm_ed(time_series)
            loss = criterion(re_con, time_series)
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
        lstm_ed.eval()
        batch_time = AverageMeter()
        val_losses = AverageMeter()
        start = time.time()
        with torch.no_grad():

            for i, time_series in enumerate(val_loader):
                time_series.to(device)
                re_con = lstm_ed(time_series)
                loss = criterion(re_con, time_series)
                val_losses.update(loss.item())
                batch_time.update(time.time() - start)
                start = time.time()
            val_mse = val_losses.avg
            if val_mse < best_loss:
                best_loss = val_mse
                epochs_since_improvement = 0
                torch.save({'epoch': epoch + 1, 'state_dict': lstm_ed.state_dict(), 'best_loss': best_loss,
                            'optimizer': optimizer.state_dict()}, os.path.join(prefix,
                                                                               str("lstm_ed_%d_%.4f.pth.tar" % (epoch,
                                                                                                        best_loss))))
            else:
                epochs_since_improvement += 1
            print('Validation: [{0}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i,
                                                                  batch_time=batch_time,
                                                                  loss=val_losses))
        save_dic['val'].append(val_losses.avg)


if __name__ == "__main__":
    USP_Training(debutanizer=True)




