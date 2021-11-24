import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import lightgbm as lgb
from sklearn.preprocessing import KBinsDiscretizer
from utils.file import *


def generate_pretrain(pre_x, pre_y):
    os.makedirs(DIR + "pretrain", exist_ok=True)
    # Note that the test here is actually divided by itself.
    # In fact, the real test is not provided by the author of this dataset
    norm_col = pre_x.columns
    norm_col = [col for col in norm_col if "bin" not in col][1:]
    pre_x[norm_col] = (pre_x[norm_col] - pre_x[norm_col].mean()) / pre_x[norm_col].std()
    pre_x = pre_x.iloc[:, 1:]
    # pre_x = (pre_x - pre_x.mean())/pre_x.std()
    total_len = pre_y.shape[0]
    cut1_y = int(total_len*train_size)
    cut1_x = cut1_y * 60
    cut2_y = cut1_y + int(total_len*val_size)
    cut2_x = cut2_y * 60
    pre_x.iloc[:cut1_x].to_csv(DIR+"pretrain/train_x.csv", index=False)
    pre_y.iloc[:cut1_y].to_csv(DIR+"pretrain/train_y.csv", index=False)
    pre_x.iloc[cut1_x:cut2_x].to_csv(DIR+"pretrain/val_x.csv", index=False)
    pre_y.iloc[cut1_y:cut2_y].to_csv(DIR+"pretrain/val_y.csv", index=False)
    pre_x.iloc[cut2_x:].to_csv(DIR+"pretrain/test_x.csv", index=False)
    pre_y.iloc[cut2_y:].to_csv(DIR+"pretrain/test_y.csv", index=False)


def generate_discrete_features(pre_x):
    """
    First of all, it is certain that not all continuous features can be discretized properly. On the one hand,
    discretization may bring information loss; on the other hand, discretization also makes features robust.
    Time series prediction is a common problem in industrial big data. In industrial process, in addition to
    the continuous features measured by sensors, there may be some discrete features from supply chain and
    manufacturing process. FM module is introduced to capture the low dimensional information of these discrete features
    """
    field_dims = [4, 5, 5, 2, 4, 20, 20, 20, 6, 5, 5, 10, 30]

    def discrete_kmeans(pre_x, strategy="kmeans", bin_dims=None):
        assert strategy in ["uniform", "quantile", "kmeans"]
        cols = pre_x.columns.tolist()[1:]  # delete the datetime
        col_leave = read_pickle_from_file("../dataset/split_features.pickle")
        cols = [col for col in cols if col in col_leave]
        for i, col in enumerate(cols):
            print("Binning: {0}".format(col))
            col_new = col + "_bin"
            kmeans_per_k = KBinsDiscretizer(n_bins=bin_dims[i], encode='ordinal', strategy=strategy)
            kmeans_per_k.fit(pre_x[[col]])
            labels = kmeans_per_k.transform(pre_x[[col]])
            pre_x[col_new] = labels
        return pre_x

    pre_xx = discrete_kmeans(pre_x, strategy="kmeans", bin_dims=field_dims)
    return pre_xx




def preprocess_and_generate():
    data_x = pd.read_csv(DIR + "data_X.csv", decimal=',')
    data_y = pd.read_csv(DIR + "data_Y.csv", decimal=',')
    submit_y = pd.read_csv(DIR+'sample_submission.csv', sep=',')
    print(data_x.columns, data_y.columns, submit_y.columns)
    print(len(data_x), len(data_y), len(submit_y))
    data_x["H_data"] = data_x["H_data"].astype("float")
    data_x["AH_data"] = data_x["AH_data"].astype("float")
    data_x["date"] = pd.to_datetime(data_x['date_time'])
    data_x1 = data_x.set_index(data_x["date"])
    data_x1.drop("date", axis=1, inplace=True)
    start = pd.to_datetime(data_y["date_time"].iloc[0]) - pd.Timedelta('0 days 01:00:00')
    end = data_y["date_time"].iloc[-1]
    # set the 2_hour_before signal to the data_x. simple set all the fake quality label. better add gaussian noise
    fake_label = np.concatenate([data_y["quality"].values.reshape(-1, 1)] * 60, axis=1).reshape(-1, 1)
    data_x1 = data_x1.loc[start:end][1:]
    data_x1["2_hours_before"] = fake_label
    return data_x1, data_y


if __name__ == "__main__":
    train_size, val_size, test_size = 0.8, 0.1, 0.1
    DIR = "../dataset/"
    pre_x, pre_y = preprocess_and_generate()
    pre_x = generate_discrete_features(pre_x)
    generate_pretrain(pre_x, pre_y)



