from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
import sys
import sklearn
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
import os
import pandas as pd
from pandas.plotting import scatter_matrix
import tarfile
import urllib
import pandas as pd
from zlib import crc32

"""Presets"""
# Ignore useless warnings (see SciPy issue #5998)
warnings.filterwarnings(action="ignore", message="^internal gelsd")
ZILLOW_PATH = os.path.join("datasets")
np.random.seed(42)

# To plot pretty figures
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images")
os.makedirs(IMAGES_PATH, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    """Function to save images"""
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def load_zillow_data(zillow_path=ZILLOW_PATH, data='zillow'):
    """Load zillow data"""
    csv_path = os.path.join(zillow_path, data+'.csv')
    return pd.read_csv(csv_path, index_col='zpid')


"""Load and prep data"""
zillow = load_zillow_data(data='Los_Angeles')
zillow.shape
zillow.columns
zillow.head()
zillow.info()
zillow.zipcode = pd.to_numeric(zillow.zipcode.str.strip())
# Drop houses with missing price values
zillow.dropna(subset=['price'], inplace=True)
zillow = zillow[zillow.price < 5000000]
# Drop wrong types of properties
zillow = zillow[zillow['homeType'].isin(['SINGLE_FAMILY', 'MULTI_FAMILY', 'CONDO'])]

"""Create some spatial plots"""
zillow.plot(kind='scatter', x='longitude', y='latitude', alpha=.1, s=zillow['livingArea']/50,
            label='Square feet', c='zestimate', cmap=plt.get_cmap('jet'), sharex=False, figsize=(10, 7), colorbar=True)


# Drop uninformative variables
zillow_prep = zillow.drop(['Unnamed: 0', 'imageLink', 'contactPhone', 'isUnmappable', 'rentalPetsFlags', 'mediumImageLink', 'hiResImageLink', 'watchImageLink', 'contactPhoneExtension', 'tvImageLink', 'tvCollectionImageLink', 'price', 'tvHighResImageLink', 'zillowHasRightsToImages', 'moveInReady', 'priceForHDP', 'title', 'group_type', 'openHouse',
                           'isListingOwnedByCurrentSignedInAgent', 'brokerId', 'grouping_name', 'priceSuffix', 'zestimate',
                           'desktopWebHdpImageLink', 'hideZestimate', 'streetAddressOnly', 'unit', 'open_house_info', 'providerListingID', 'newConstructionType', 'datePriceChanged', 'dateSold', 'streetAddress', 'city', 'state', 'timeOnZillow', 'currency', 'country', 'priceChange', 'isRentalWithBasePrice',
                           'isListingClaimedByCurrentSignedInUser', 'lotId', 'lotId64', 'shouldHighlight', 'isPreforeclosureAuction', 'grouping_id', 'comingSoonOnMarketDate', 'url'], axis=1)
zillow_prep.shape
# Numerical variables
num_cols = list(zillow_prep.select_dtypes(include=['float', 'int64']).drop(
    ['videoCount', 'zipcode', 'yearBuilt'], axis=1).columns)

# Set negative numerical values (excluding lat/long) to missing
for col in num_cols[2:]:
    zillow_prep.loc[(zillow_prep[col] < 0), col] = np.nan

# Plot the numerical variables before log transformation
zillow_prep[num_cols].hist(bins=50, figsize=(20, 15))
# save_fig("attribute_histogram_plots")

# Log transform skewed variables
for col in num_cols:
    span = zillow_prep[col].max()-zillow_prep[col].min()
    if span > 150:
        x = zillow_prep[col]
        x[x == 0] = .00001
        x = np.log10(x)
        zillow_prep[col] = x

# Plot the numerical variables again after log transformation
zillow_prep[num_cols].hist(bins=50, figsize=(20, 15))
# save_fig("attribute_histogram_plots")

## Categorical variables ##
# First, clean up yearBuilt
zillow_prep.yearBuilt.replace(-1, np.nan, inplace=True)
zillow_prep['quants'] = pd.qcut(zillow_prep.yearBuilt, [0, .25, .5, .75, 1], labels=[1, 2, 3, 4])
zillow_prep.pivot_table(values='yearBuilt', index=['quants'], aggfunc=[np.mean, 'count'])
zillow_prep.drop('yearBuilt', inplace=True, axis=1)

# List cat_cols
cat_cols = list(zillow_prep.drop(num_cols, axis=1).columns)
zillow_prep[cat_cols].info()

# First, recode and fix missingness
zillow_prep.loc[zillow_prep['listing_sub_type'].str.contains('FSBO'), 'listing_sub_type'] = 1
zillow_prep.loc[zillow_prep['listing_sub_type'] != 1, 'listing_sub_type'] = 0
zillow_prep.loc[zillow_prep['isNonOwnerOccupied'] != 1, 'isNonOwnerOccupied'] = 0
zillow_prep.loc[zillow_prep['priceReduction'].isnull() == 1, 'priceReduction'] = 0
zillow_prep.loc[zillow_prep['priceReduction'] != 0, 'priceReduction'] = 1
zillow_prep.loc[zillow_prep['videoCount'].isnull() == 1, 'videoCount'] = 0
zillow_prep.loc[zillow_prep['videoCount'] != 0, 'videoCount'] = 1
zillow_prep[cat_cols].info()

# Cast categorical columns to strings prior to onehot transformation
le = LabelEncoder()
zillow_prep[cat_cols] = zillow_prep[cat_cols].applymap(str)

## Build a pipeline ##
num_pipeline = Pipeline(steps=[
    ('imputer', KNNImputer()),
    ('t', PolynomialFeatures(degree=2)),
    ('std_scaler', StandardScaler())])

cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Instantiate a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)])

# Create a train test split
X = zillow_prep
y = zillow.price
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
X_train.shape

# First, check the linear model
OLS = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', LinearRegression())])
OLS.fit(X_train, y_train)
pred = OLS.predict(X_train)
np.sqrt(mean_squared_error(y_train, pred))


# Parameter grids
ridge_grid = {'regressor__alpha': np.arange(0.5, 5, 0.5)}

lasso_grid = {'regressor__alpha': np.arange(5, 50, 5)}

enet_grid = {"regressor__alpha": np.arange(.008, .02, .001),
                      "regressor__l1_ratio": np.arange(0.0, 0.5, 0.05),
                      "regressor__max_iter": np.arange(20, 200, 20),
                      }

svm_grid =
    # First try linear and rbf kernels
    [{'regressor__C': np.arange(1300, 1600, 25),
              'regressor__gamma': np.arange(1, 10, 1),
              'regressor__kernel': ['linear', 'rbf']},
    # First try linear and rbf kernels
    {'regressor__C': np.arange(100, 1500, 100),
              'regressor__gamma': np.arange(1, 10, 1),
              'regressor__kernel': ['poly'],
              'regressor_degree': [1, 2, 3, 4]}]


rf_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'regressor__n_estimators': np.arange(
        10, 100, 10), 'regressor__max_features': np.arange(5, 50, 5)},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'regressor__bootstrap': [False], 'regressor__n_estimators': np.arange(
        5, 15, 1), 'regressor__max_features': np.arange(5, 50, 5)},
]

# Models to compare
models = [
    [Ridge(), ridge_grid],
    [Lasso(), lasso_grid],
    [ElasticNet(), enet_grid],
    [SVR(), svm_grid],
    [RandomForestRegressor(), rf_grid],
]

# Use GridSearchCV to assess alterantive models
results = {}
for model, grid in models:
    estimator = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    CV = GridSearchCV(estimator, scoring='neg_root_mean_squared_error',
                            param_grid=grid, cv=3)
    results[type(model).__name__] = CV.fit(X_train, y_train)

for key in results.keys():
    print(key, ': ', results[key].best_params_)


def RMSE(model, log=False):

"""Calculate the RMSE from a model"""
  pred = model.predict(X_test)
   if log == True:
        pred = 10 ** pred
    mse = mean_squared_error(y_test, pred)
    rmse = np.round(np.sqrt(mse), decimals=3)
    print(rmse)


# Examine predicted versus actual labels
plt.scatter(y_test, y_pred, marker='o', alpha=.1, c=y_test, cmap=plt.get_cmap('jet'))


def plot_learning_curves(model, X, y):
    """Plot the learning curve"""
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend(loc="upper right", fontsize=14)   # not shown in the book
    plt.xlabel("Training set size", fontsize=14)  # not shown
    plt.ylabel("RMSE", fontsize=14)              # not shown


plot_learning_curves(lin_reg, X_prep, y)
