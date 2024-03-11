import pandas as pd
from copy import deepcopy
from catboost import CatBoostRanker, Pool
import numpy as np
import json 



pd.set_option('display.max_columns', None)
test_data = pd.read_csv('test_df.csv')
train_data = pd.read_csv('train_df.csv')

train_data = train_data.drop(['feature_73','feature_74','feature_75'], axis=1)
test_data = test_data.drop(['feature_73','feature_74','feature_75'], axis=1)

X_train = train_data.drop(['search_id'], axis=1).values
y_train = train_data['target'].values
queries_train = train_data['search_id'].values


X_test = test_data.drop(['search_id'], axis=1).values
y_test = test_data['target'].values
queries_test = test_data['search_id'].values

max_relevance = np.max(y_train)
y_train = y_train.astype(np.int64) / max_relevance
y_test = y_test.astype(np.int64) / max_relevance

y_train = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)

train = Pool(
    data=X_train,
    label=y_train,
    group_id=queries_train
)

test = Pool(
    data=X_test,
    label=y_test,
    group_id=queries_test
)

default_parameters = {
    'iterations': 2000,
    'custom_metric': ['NDCG', 'PFound', 'AverageGain:top=10'],
    'verbose': False,
    'random_seed': 0,
}

parameters = {}

def fit_model(loss_function, additional_params=None, train_pool=train, test_pool=test):
    parameters = deepcopy(default_parameters)
    parameters['loss_function'] = loss_function
    parameters['train_dir'] = loss_function
    
    if additional_params is not None:
        parameters.update(additional_params)
        
    model = CatBoostRanker(**parameters)
    model.fit(train_pool, eval_set=test_pool, plot=True)
    
    return model

model = fit_model('PairLogitPairwise', {'custom_metric': ['NDCG']})

test_data['predictions'] = model.predict(test)

metrics = dict()
metrics['ndsg_train_score'] = model.score(X_train, y_train, queries_train)
metrics['ndsg_test_score'] = model.score(X_test, y_test, queries_test)

model.save_model('catboost_model.bin')

with open("metrics.json", "w") as outfile: 
    json.dumps(metrics, outfile)
