# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 10:53:19 2022

@author: Yazka
"""
#1
import os
#2
import tarfile
from six.moves import urllib
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
#3
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
#4
fetch_housing_data()
#5
import pandas as pd
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
#6
housing = load_housing_data()
#7
housing.head(10)
#8
housing.info()
#9
housing["ocean_proximity"].value_counts()
#10
housing.describe()
#11
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()
#12
import numpy as np
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
#13
train_set, test_set = split_train_test(housing, 0.2)
#14
len(train_set)
#15
housing["income_cat"] = pd.cut(housing["median_income"],
bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
labels=[1, 2, 3, 4, 5])
#16
housing["income_cat"].hist()
#17
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
#18
strat_test_set["income_cat"].value_counts() / len(strat_test_set)
#19
def functie ( a , b ) :
    c = a + 1 
    d = b + 2
    return c , d 
#20
z=functie(1,2)
#21
z
#22
corr_matrix = housing.corr()
#23
corr_matrix["median_house_value"].sort_values(ascending=False)
#24
from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms","housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
#25
housing.plot(kind="scatter", x="median_income", y="median_house_value",alpha=0.1)
#26
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
#27
housing.head(10)
#28
corr_matrix = housing.corr()
#29
corr_matrix["median_house_value"].sort_values(ascending=False)
#30
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
#31
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
#32
housing["income_cat"] = pd.cut(housing["median_income"],
bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
labels=[1, 2, 3, 4, 5])
#32
import numpy as np
#33
strat_train_set
#34
housing_labels
#35
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
#36
housing_num = housing.drop("ocean_proximity", axis=1)
#37
housing_num.head()
#38
imputer.fit(housing_num)
#39
imputer.statistics_
#40
housing_num.median().values
#41
X = imputer.transform(housing_num)
X
#42
housing_tr = pd.DataFrame(X, columns=housing_num.columns)
#43
imputer.strategy
#44
housing_cat = housing[["ocean_proximity"]]
#45
housing_cat.head(10)
#46
from sklearn.preprocessing import OrdinalEncoder
#47
ordinal_encoder = OrdinalEncoder()
#48
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
#49
housing_cat_encoded[:10]
#50
ordinal_encoder.categories_
#51
from sklearn.preprocessing import OneHotEncoder
#52
cat_encoder = OneHotEncoder()
#53
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
#54
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
#55
housing_cat_1hot.toarray()
#56
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self # nothing else to do
    
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
#57
housing_extra_attribs
#58
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)
#59
from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

housing_prepared = full_pipeline.fit_transform(housing)
#60
housing_prepared[1,]
#61
housing_labels[1]
#62
housing_num.head(5)
#63
cat_attribs
#64
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
#65
some_data = housing.iloc[:5]
#66
some_data
#67
some_labels = housing_labels.iloc[:5]
#68
some_labels
#69
some_data_prepared = full_pipeline.transform(some_data)
#70
print("Predictions:", lin_reg.predict(some_data_prepared))
#71
from sklearn.metrics import mean_squared_error
#72
housing_predictions = lin_reg.predict(housing_prepared)
#73
lin_mse = mean_squared_error(housing_labels, housing_predictions)
#74
lin_rmse = np.sqrt(lin_mse)
#75
lin_rmse
#76
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
#77
housing_predictions = tree_reg.predict(housing_prepared)
#78
tree_mse = mean_squared_error(housing_labels, housing_predictions)
#79
tree_rmse = np.sqrt(tree_mse)
#80
tree_rmse
#81
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
#82
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
#83
display_scores(tree_rmse_scores)
#84
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
#85
lin_rmse_scores = np.sqrt(-lin_scores)
#86
display_scores(lin_rmse_scores)
#87
from sklearn.ensemble import RandomForestRegressor
#88
forest_reg = RandomForestRegressor()
#89
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
#90
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
#91
forest_rmse_scores = np.sqrt(-forest_scores)
#92
display_scores(forest_rmse_scores)
#93
housing_predictions = forest_reg.predict(housing_prepared)
#94
forest_rmse = np.sqrt(forest_rmse)
#95
forest_rmse
#96
from sklearn.model_selection import GridSearchCV
param_grid = [
{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
{'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]
#97
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)
#98
grid_search.best_estimator_
#99
grid_search.best_params_
#100
cvres = grid_search.cv_results_
#101
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
    #102
feature_importances = grid_search.best_estimator_.feature_importances_
#103
feature_importances
#104
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
#104
cat_encoder = full_pipeline.named_transformers_["cat"]
#105
cat_one_hot_attribs = list(cat_encoder.categories_[0])
#106
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
#107
sorted(zip(feature_importances, attributes), reverse=True)
#108
final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse) # => evaluates to 47,730.2
#109
final_rmse
#Grid Search
from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)
GridSearchCV(cv=5, estimator=RandomForestRegressor(random_state=42),
             param_grid=[{'max_features': [2, 4, 6, 8],
                          'n_estimators': [3, 10, 30]},
                         {'bootstrap': [False], 'max_features': [2, 3, 4],
                          'n_estimators': [3, 10]}],
             return_train_score=True, scoring='neg_mean_squared_error')
grid_search.best_params_
grid_search.best_estimator_
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
pd.DataFrame(grid_search.cv_results_)
#Randomized Search
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_scvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

#1
from sklearn.model_selection import GridSearchCV

param_grid = [
        {'kernel': ['linear'], 'C': [10., 30., 100., 300., 1000., 3000., 10000., 30000.0]},
        {'kernel': ['rbf'], 'C': [1.0, 3.0, 10., 30., 100., 300., 1000.0],
         'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
svm_reg = SVR()
grid_search = GridSearchCV(svm_reg, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)
grid_search.fit(housing_prepared, housing_labels)
negative_mse = grid_search.best_score_
rmse = np.sqrt(-negative_mse)
rmse
grid_search.best_params_

#3
from sklearn.base import BaseEstimator, TransformerMixin

def indices_of_top_k(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])

class TopFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k
    def fit(self, X, y=None):
        self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)
        return self
    def transform(self, X):
         return X[:, self.feature_indices_]
    k = 5
    top_k_feature_indices = indices_of_top_k(feature_importances, k)
    top_k_feature_indices
    np.array(attributes)[top_k_feature_indices]
    sorted(zip(feature_importances, attributes), reverse=True)[:k]
    preparation_and_feature_selection_pipeline = Pipeline([
    ('preparation', full_pipeline),
    ('feature_selection', TopFeatureSelector(feature_importances, k))
])
    housing_prepared_top_k_features = preparation_and_feature_selection_pipeline.fit_
    housing_prepared_top_k_features[0:3]
    housing_prepared[0:3, top_k_feature_indices]
    