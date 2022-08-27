"""
@version: python3.6
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc
from src.clust.score import make_dirs

## load data
# data_dir = '/data1/tsq/contrastive/clust_documents/animal/ensemble/r2_recall_regression/few_shot500/'
data_dir = '/data1/tsq/contrastive/clust_documents/animal/ensemble/f1_regression/few_shot500/'
train_path = os.path.join(data_dir, 'train.csv')
test_path = os.path.join(data_dir, 'test.csv')
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)
num_round = 1000
section_num = 10
addition_inverse_num = 4

## category feature, -1 means test
test_data['split'] = -1
data = pd.concat([train_data, test_data])
topic_feature = [f'topic_sec{i}' for i in range(section_num)]
inverse_feature = [f'inverse{i}' for i in range(addition_inverse_num + 1)]
none_feature = ['none_prompt']

""" LabelEncoder is for discrete data
cate_feature = []
for item in cate_feature:
    data[item] = LabelEncoder().fit_transform(data[item])
"""

train = data[data['split'] != -1]
test = data[data['split'] == -1]

# Clean up the memory
del data, train_data, test_data
gc.collect()

## get train feature
# del_feature = ['label', 'split', 'unnamed:0', 'sent_id', 'data_id']
# features = [i for i in train.columns if i not in del_feature]
features = topic_feature + inverse_feature + none_feature

train_x = train[features]
train_y = train['label'].astype(float)
test_x = test[features]
test_y = test['label'].astype(float)

##train and predict
params = {'num_leaves': 38,
          'min_data_in_leaf': 50,
          'objective': 'regression',
          'max_depth': -1,
          'learning_rate': 0.02,
          "min_sum_hessian_in_leaf": 6,
          "boosting": "gbdt",
          "feature_fraction": 0.9,
          "bagging_freq": 1,
          "bagging_fraction": 0.7,
          "bagging_seed": 11,
          "lambda_l1": 0.1,
          "verbosity": -1,
          "nthread": 4,
          'metric': 'mae',
          "random_state": 1453,
          # 'device': 'gpu'
          }


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true))) * 100


def smape_func(preds, dtrain):
    label = dtrain.get_label().values
    epsilon = 0.1
    summ = np.maximum(0.5 + epsilon, np.abs(label) + np.abs(preds) + epsilon)
    smape = np.mean(np.abs(label - preds) / summ) * 2
    return 'smape', float(smape), False


folds = KFold(n_splits=5, shuffle=True, random_state=1453)
oof = np.zeros(train_x.shape[0])
predictions = np.zeros(test_x.shape[0])

# train_y = np.log1p(train_y)  # Data smoothing
feature_importance_df = pd.DataFrame()
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_x)):
    print("fold {}".format(fold_ + 1))
    trn_data = lgb.Dataset(train_x.iloc[trn_idx], label=train_y.iloc[trn_idx])
    val_data = lgb.Dataset(train_x.iloc[val_idx], label=train_y.iloc[val_idx])

    clf = lgb.train(params,
                    trn_data,
                    num_round,
                    valid_sets=[trn_data, val_data],
                    verbose_eval=200,
                    # categorical_feature=cate_feature,
                    early_stopping_rounds=200)
    oof[val_idx] = clf.predict(train_x.iloc[val_idx], num_iteration=clf.best_iteration)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    predictions += clf.predict(test_x, num_iteration=clf.best_iteration) / folds.n_splits

make_dirs(os.path.join(data_dir, 'lgb'))

id_cols = ['data_id', 'sent_id']
ids = train[id_cols]
oof_df = pd.DataFrame(oof, columns=['pred_score'])
train_csv = pd.concat([ids, oof_df], axis=1, ignore_index=True)
train_csv.columns = id_cols + ['pred_score']
train_csv.to_csv(os.path.join(data_dir, 'lgb', "pred_train.csv"), index=False)
print('train mse %.6f' % mean_squared_error(train_y, oof))
print('train mae %.6f' % mean_absolute_error(train_y, oof))

# result = np.expm1(predictions)  # reduction

result = predictions
result_df = pd.DataFrame(result, columns=['pred_score'])
test_csv = pd.concat([test[id_cols], result_df], axis=1, ignore_index=True)
test_csv.columns = id_cols + ['pred_score']
test_csv.to_csv(os.path.join(data_dir, 'lgb', "pred_test.csv"), index=False)
print('test mse %.6f' % mean_squared_error(test_y, result))
print('test mae %.6f' % mean_absolute_error(test_y, result))

## plot feature importance
cols = (feature_importance_df[["Feature", "importance"]].groupby("Feature").mean().sort_values(by="importance",
                                                                                               ascending=False).index)
best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)].sort_values(by='importance',
                                                                                                ascending=False)
plt.figure(figsize=(8, 10))
sns.barplot(y="Feature",
            x="importance",
            data=best_features.sort_values(by="importance", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('/home/tsq/TopCLS/src/statistics/regression/lgb_importances.png')
