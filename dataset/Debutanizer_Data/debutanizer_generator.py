import numpy as np
import pandas as pd


def Preprocess_Debutanizer(step=2):
    data = pd.read_csv('Debutanizer_Data.txt', sep='\s+')
    data = data.values

    # 数据转化:
    x_temp = data[:, :7]
    y_temp = data[:, 7]

    x_new = np.zeros([2390, 13])
    x_6 = x_temp[:, 4]
    x_9 = (x_temp[:, 5] + x_temp[:, 6])/2
    x_new[:, :5] = x_temp[4: 2394, :5]

    x_new[:, 5] = x_6[3: 2393]
    x_new[:, 6] = x_6[2: 2392]
    x_new[:, 7] = x_6[1: 2391]
    x_new[:, 8] = x_9[4: 2394]


    x_new[:, 9] = y_temp[3: 2393]
    x_new[:, 10] = y_temp[2: 2392]
    x_new[:, 11] = y_temp[1:2391]
    x_new[:, 12] = y_temp[:2390]
    y_new = y_temp[4: 2394]
    y_new = y_new.reshape([-1, 1])

    #Split the Dataset in GSTAE setting
    train_x = x_new[:1000, :]
    train_y = y_new[:1000]
    pd.DataFrame(train_x, columns=np.arange(train_x.shape[1])).to_csv("train_x.csv", index=False)
    pd.DataFrame(train_y, columns=np.arange(train_y.shape[1])).to_csv("train_y.csv", index=False)

    x_validation = x_new[1000-step:1300, :]
    y_validation = y_new[1000-step:1300]
    pd.DataFrame(x_validation, columns=np.arange(x_validation.shape[1])).to_csv("val_x.csv", index=False)
    pd.DataFrame(y_validation, columns=np.arange(y_validation.shape[1])).to_csv("val_y.csv", index=False)

    test_x = x_new[1300-step:1804, :]
    test_y = y_new[1300-step:1804]
    pd.DataFrame(test_x, columns=np.arange(test_x.shape[1])).to_csv("test_x.csv", index=False)
    pd.DataFrame(test_y, columns=np.arange(test_y.shape[1])).to_csv("test_y.csv", index=False)

    return 0


if __name__ == "__main__":
    Preprocess_Debutanizer(step=2)