import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import lightgbm as lgb


DIR = "../dataset/"

data_x = pd.read_csv(DIR + "data_X.csv", decimal=',')
data_y = pd.read_csv(DIR + "data_Y.csv", decimal=',')
submit_y = pd.read_csv(DIR+'sample_submission.csv', sep=',')
print(data_x.columns, data_y.columns, submit_y.columns)
print(len(data_x), len(data_y), len(submit_y))

train_df = data_x.merge(data_y, left_on='date_time', right_on='date_time')
test_df = data_x.merge(submit_y, left_on='date_time', right_on='date_time').drop("quality", axis=1)

train_df.drop("date_time", axis=1, inplace=True)
train_df["H_data"] = train_df["H_data"].astype("float")
train_df["AH_data"] = train_df["AH_data"].astype("float")
train_df["2_hours_before"] = [train_df["quality"].mean()]*2 + train_df["quality"][:-2].values.tolist()

# the code below is mainly modified from  https://www.kaggle.com/ogrellier/feature-selection-with-null-importances


def get_feature_importances(data, shuffle, seed=None):
    # Gather real features
    train_features = [f for f in data if f not in ['quality']]
    # Go over fold and keep track of CV score (train and valid) and feature importances

    # Shuffle target if required
    y = data['quality'].copy()
    if shuffle:
        # Here you could as well use a binomial distribution
        y = data['quality'].copy().sample(frac=1.0)

    # Fit LightGBM in RF mode, yes it's quicker than sklearn RandomForest
    dtrain = lgb.Dataset(data[train_features].values, y.values, free_raw_data=False, silent=True)
    lgb_params = {
        'objective': 'regression',
        'boosting_type': 'rf',
        'subsample': 0.623,
        'colsample_bytree': 0.7,
        'num_leaves': 127,
        'max_depth': 8,
        'seed': seed,
        'bagging_freq': 1,
        'n_jobs': 4,

    }

    # Fit the model
    clf = lgb.train(params=lgb_params, train_set=dtrain, num_boost_round=200)

    # Get feature importances
    imp_df = pd.DataFrame()
    imp_df["feature"] = list(train_features)
    imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')
    imp_df["importance_split"] = clf.feature_importance(importance_type='split')
    imp_df['trn_score'] = np.sqrt(mean_squared_error(y, clf.predict(data[train_features])))

    return imp_df


# Seed the unexpected randomness of this world
np.random.seed(123)
# Get the actual importance, i.e. without shuffling
actual_imp_df = get_feature_importances(data=train_df, shuffle=False)

print("Actual importance distribution: ")
print(actual_imp_df.head())

null_imp_df = pd.DataFrame()
nb_runs = 200

start = time.time()
dsp = ''
for i in range(nb_runs):
    # Get current run importances
    imp_df = get_feature_importances(data=train_df, shuffle=True)
    imp_df['run'] = i + 1
    # Concat the latest importances with the old ones
    null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)
    # Erase previous message
    for l in range(len(dsp)):
        print('\b', end='', flush=True)
    # Display current run and time used
    spent = (time.time() - start) / 60
    dsp = 'Done with %4d of %4d (Spent %5.1f min)' % (i + 1, nb_runs, spent)
    print(dsp, end='', flush=True)

print("NUll importance distribution: ")
print(null_imp_df.head())

print("Save null importance and actual importance distribution!")
null_imp_df.to_csv("../dataset/null_importances_distribution_rf.csv", index=False)
actual_imp_df.to_csv("../dataset/actual_importances_distribution_rf.csv", index=False)
