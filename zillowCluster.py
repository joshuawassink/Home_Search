from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import palettable
from matplotlib.colors import ListedColormap
import sys
import sklearn
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import urllib
from zlib import crc32
import seaborn as sns

"""Presets"""
warnings.filterwarnings(action="ignore", message="^internal gelsd")
# Where to save the figures
PROJECT_ROOT_DIR = "."
ZILLOW_PATH = os.path.join(PROJECT_ROOT_DIR, "datasets")
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images")
os.makedirs(IMAGES_PATH, exist_ok=True)
np.random.seed(42)


def printText(text):
    """Print formatted notes

    Args:
        text (str): a text string

    Returns:
        Formatted printout of text
    """
    print('-'*100)
    print(text)
    print('-'*100)


# To plot pretty figures
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
# Set styles
styles = ['o', 's', 'x', 'o', '+', '*', 'X', 'o', '^', 'D', '4']
plt.style.use('seaborn-dark')


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    """Function to save images"""
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def load_zillow_data(data, zillow_path=ZILLOW_PATH, from_csv=True):
    """Load housing data

    Args:
        data (list): list of dicts to convert to pandas df
        zillow_path (str): string formatted path to file directory
        from_csv (bool): default = True

    Returns:
        Pandas DF with instances of individual listings and features of those listings

    """
    if from_csv:
        csv_path = os.path.join(zillow_path, data+'.csv')
        df = pd.read_csv(csv_path, index_col='zpid')
    else:
        df = pd.DataFrame(data, index_col='zpid')
    # Convert zipcode from string to numeric
    df.zipcode = pd.to_numeric(df.zipcode.str.strip())
    df = df[df['homeType'].isin(['SINGLE_FAMILY', 'MULTI_FAMILY', 'CONDO'])]

    # Drop outliers
    df = df[df.price < 2000000]
    df = df[df.lotSize < 1000000]
    df = df[df.livingArea < 10000]
    # Drop uninformative variables
    df = df.drop(['Unnamed: 0', 'imageLink', 'contactPhone', 'isUnmappable', 'rentalPetsFlags', 'mediumImageLink', 'hiResImageLink', 'watchImageLink', 'contactPhoneExtension', 'tvImageLink', 'tvCollectionImageLink', 'tvHighResImageLink', 'zillowHasRightsToImages', 'moveInReady', 'priceForHDP', 'title', 'group_type', 'openHouse',
                  'isListingOwnedByCurrentSignedInAgent', 'brokerId', 'grouping_name', 'priceSuffix', 'listing_sub_type',
                  'desktopWebHdpImageLink', 'hideZestimate', 'streetAddressOnly', 'unit', 'open_house_info', 'providerListingID', 'isZillowOwned', 'homeStatusForHDP', 'newConstructionType', 'datePriceChanged', 'dateSold', 'streetAddress', 'city', 'state', 'timeOnZillow', 'isNonOwnerOccupied', 'currency', 'country', 'priceChange', 'isRentalWithBasePrice',
                  'isListingClaimedByCurrentSignedInUser', 'lotId', 'lotId64', 'shouldHighlight', 'isPreforeclosureAuction', 'grouping_id', 'comingSoonOnMarketDate', 'url'], axis=1)
    # Prep numerical variables
    num_cols = list(df.select_dtypes(include=['float', 'int64']).drop(
        ['videoCount', 'zipcode'], axis=1).columns)
    # Set negative numerical values (excluding lat/long) to missing
    for col in num_cols[2:]:
        df.loc[(df[col] < 0), col] = np.nan
    # Log transform skewed variables
    logged = []
    for col in num_cols:
        span = df[col].max()-df[col].min()
        if span > 500:
            logged.append(col)
            x = df[col].copy()
            x[x == 0] = .00001
            x = np.log10(x)
            df[col] = x.copy()

    # Prep categorical variables
    cat_cols = list(df.drop(num_cols, axis=1).columns)
    # Recode and fix missingness
    df.loc[df['priceReduction'].isnull() == 1, 'priceReduction'] = 0
    df.loc[df['priceReduction'] != 0, 'priceReduction'] = 1
    df.loc[df['videoCount'].isnull() == 1, 'videoCount'] = 0
    df.loc[df['videoCount'] != 0, 'videoCount'] = 1
    # Cast categorical columns to strings prior to onehot transformation
    le = LabelEncoder()
    df[cat_cols] = df[cat_cols].applymap(str)
    printText('Data contain {} instances with {} features'.format(df.shape[0], df.shape[1]))
    return df, num_cols, cat_cols, logged


def data_prep(num_cols, cat_cols, df):
    """Load and prep data

    Args:
        num_cols (list): list of numeric columns
        cat_cols (list): list of categorical columns

    Returns:
        X (matrix): Matrix of preprocessed features
    """
    # Build a preprocessing pipeline for numeric variables
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('std_scaler', StandardScaler())])
    # Build a preprocessing pipeline for categorical variables
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    # Instantiate a preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols)])
    # Generate the preprocessor
    X = preprocessor.fit_transform(df)
    return X


def K_cluster(X):
    """Fit and assess K-means clustering algorithm

    Args:
        X (Matrix): Transformed

    Returns:
        kmeans (kmeans object): preferred kmeans solution
        precipice (int): preferred number of of clusters
    """
    # Fit kmeans algorithm for 1-10 clusters
    printText('Fitting KMeans algorithm for 1-10 possible clusters. This may take a while')
    kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X)
                    for k in range(1, 10)]
    # Extract intertias
    inertias = [model.inertia_ for model in kmeans_per_k]
    # Extract silhouettes
    silhouette_scores = [silhouette_score(X, model.labels_)
                         for model in kmeans_per_k[1:]]
    # Determine the preferred number of clusters based on the silhoutette score
    precipice = 1
    max_decline = 0
    for index, silhouette in enumerate(silhouette_scores[:7]):
        decline = silhouette-silhouette_scores[index+1]
        if decline > max_decline:
            max_decline = decline
            precipice = index+2
    printText('The silhouette precipice suggests an optimal solution with {} clusters'.format(
        precipice))

    # Plot the inertia
    plt.figure(figsize=(8, 3))
    plt.plot(range(1, 10), inertias, "bo-")
    plt.xlabel("$k$", fontsize=14)
    plt.ylabel("Inertia", fontsize=14)
    plt.annotate('Elbow',
                 xy=(precipice, inertias[precipice-1]),
                 xytext=(0.55, 0.55),
                 textcoords='figure fraction',
                 fontsize=16,
                 arrowprops=dict(facecolor='black', shrink=0.1)
                 )
    plt.axis([1, 8.5, 30000, 60000])
    save_fig('intertia_plot')
    plt.show()

    # Plot the silhouette score
    plt.figure(figsize=(8, 3))
    plt.plot(range(2, 10), silhouette_scores, "bo-")
    plt.xlabel("$k$", fontsize=14)
    plt.ylabel("Silhouette score", fontsize=14)
    plt.axis([1.8, 9.2, 0.12, 0.2])
    save_fig('silhouette_plot')
    plt.show()
    return kmeans_per_k[precipice-1], precipice


def K_plots(df, logged, kmeans, num_cols, precipice):
    # Copy df for plotting and summarizing
    df_plot = df.copy()
    # Undo natural logs
    for col in df_plot[logged]:
        df_plot[col] = 10**df_plot[col]
    # Combine non-single family homes
    df_plot.loc[df_plot['homeType'] == 'MULTI_FAMILY', 'homeType'] = 'CONDO'
    # Add labels to dataframe
    df_plot['labels'] = kmeans.labels_ + 1

    # Summarize the clusters
    K_summary = df_plot.groupby('labels')[num_cols].mean().T.round(1)
    print(K_summary.to_markdown(tablefmt="grid"))

    ## Plot the clusters##
    # Transform price and square footage
    df_plot['price'] = df_plot['price']/1000
    # Add tick marks
    tickMarks = [i for i in range(1, precipice+1)]
    # Create scatter function

    def scatter(x, y, xlab, ylab):
        plt.figure(figsize=(10, 7))
        for i, cat in enumerate(df_plot.homeType.unique()):
            sc = plt.scatter(df_plot.loc[df_plot['homeType'] == cat, x], df_plot.loc[df_plot['homeType'] == cat, y], label=cat,
                             alpha=.2, c=df_plot.loc[df_plot['homeType'] == cat, 'labels'], marker=styles[i],
                             cmap=ListedColormap(palettable.cartocolors.qualitative.Bold_6.mpl_colors))
        plt.colorbar(sc, values=tickMarks, ticks=tickMarks, label='Clusters')
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.legend(fontsize=10)
        plt.title('Kmeans Clusters by lat/long')
        save_fig('Kmeans_lat_long_plot')
        plt.show()

    scatter('latitude', 'longitude', 'Latitude', 'Longitude')
    scatter('price', 'livingArea', 'Home Price ($1,000s)', 'Home Size (SqFt)')


def viewClusters(data, from_csv=True):
    """Identify, describe and visualize clusters in the listing data

    Args:
        data (df): A dataframe object

    Returns:
        None

    """
    # Load listing data
    df, num_cols, cat_cols, logged = load_zillow_data(data, from_csv)
    X = data_prep(num_cols, cat_cols, df=df)
    kmeans, precipice = K_cluster(X)
    K_plots(df=df, logged=logged, kmeans=kmeans, num_cols=num_cols, precipice=precipice)
