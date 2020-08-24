import io
import tailer as tl
from lxml import html
import requests
import unicodecsv as csv
import json
from urllib.request import Request, urlopen
import pandas as pd
import math
import numpy as np
from time import sleep
import zipcodes
import os
import warnings
import matplotlib as mpl
from pandas.plotting import scatter_matrix
import tarfile
import urllib
import pandas as pd
from zlib import crc32
from sklearn.model_selection import train_test_split
os.getcwd()
"""Presets"""
# Ignore useless warnings (see SciPy issue #5998)
warnings.filterwarnings(action="ignore", message="^internal gelsd")
ZILLOW_PATH = os.path.join("datasets", "zillow")
os.makedirs(ZILLOW_PATH, exist_ok=True)  # Create data repository if necessary
np.random.seed(42)


def save_csv(df, csv_id):
    path = os.path.join(ZILLOW_PATH, csv_id + '.csv')
    if not os.path.exists:
        print('Saving new CSV', csv_id)
        # Write the dataframe to CSV
        df.to_csv(path, index=True)
    else:
        print('Merging data with existing CSV', csv_id)
        # Load Current CSV
        existing = pd.read_csv(path, index_col='zpid')
        # Merge using combine_first to overwrite existing cells with updated values
        df = df.combine_first(existing)
        # Write the dataframe to CSV
        df.to_csv(path, index=True)


def clean(text):
    if text:
        return ' '.join(' '.join(text).split())
    return None


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


def cityState(zip):
    city = zipcodes.matching(zip)[0]['city'].lower().replace(' ', '-')
    state = zipcodes.matching(zip)[0]['state'].lower()
    city_state = city+'-'+state
    return city_state


cityState('90015')


def create_url(zipcode, firstPage, i=None):
    if firstPage:
        # Create first Zillow URL
        url = "https://www.zillow.com/homes/for_sale/{0}_rb/?fromHomePage=true&shouldFireSellPageImplicitClaimGA=false&fromHomePageTab=buy".format(
            zipcode)
    else:
        city_state = cityState(zipcode)
        url = '{}/{}-{}/{}{}'.format('https://www.zillow.com', city_state, zipcode, i, '_p/')
    return url


def save_csv(df, csv_id):
    path = os.path.join(ZILLOW_PATH, csv_id + '.csv')
    print('Saving CSV', csv_id)
    df.to_csv(path, index=True)


def home_url(df):
    df['base'] = 'https://www.zillow.com/homedetails/'
    df['addr'] = df.streetAddress.str.replace(' ', '-')
    df['url'] = df['base'].str.cat(df['addr'])
    df['citySt'] = '-Los-Angeles-CA'
    df['suffix'] = '_zpid/'
    df['url'] = df['url'].str.cat(df['citySt'])
    df['url'] = df['url'].str.cat(df['zipcode'], sep='-')
    df['url'] = df['url'].str.cat(df['zpid'].astype(str), sep='/')
    url = df['url'].str.cat(df['suffix'])
    df.drop(['base', 'addr', 'url', 'citySt', 'suffix'], axis=1, inplace=True)
    return url


def get_url_pages(zipcode, firstPage=True, i=None):
    """Get the url, pages, and parser

        Args:
            zipcode (str): quoted 5-digit zipcode
            firstPage (Bool): optional arg in case not first page
            i (int): optional arg to format later page urls

        Returns:
            if firstPage: [url, pages, parser]
            Else: [url, parser]
    """
    url = create_url(zipcode, firstPage, i=i)
    print(url)
    # Request the url data
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    # Open and read the data
    webpage = urlopen(req).read()
    # parse the data
    parser = html.fromstring(webpage)
    if firstPage:
        """This section checks the number of results
        to determine if multiple pages need to be scraped"""
        # Get the result-count span object
        result_span = parser.xpath('//span[@class="result-count"]')
        # Extract the number of results as an integer
        results = int(result_span[0].text.split(' ')[0].replace(',', ''))
        # Number of results pages (rounded up results/40)
        pages = math.ceil(results/40)
        printText('There are currently {} properties across {} pages of results listed in zipcode {}'.format(
            results, pages, zipcode))
        return [url, pages, parser]
    else:
        return [url, parser]


def get_jason(parser):
    raw_json_data = parser.xpath(
        '//script[@data-zrr-shared-data-key="mobileSearchPageStore"]//text()')
    # Remove formatting strings
    cleaned_data = clean(raw_json_data).replace('<!--', "").replace("-->", "")
    # Convert data to json format
    json_data = json.loads(cleaned_data)
    # Extract data from the results list
    try:
        jason_data = json_data['searchResults']['listResults']
        print(1)
    except KeyError:
        jason_data = json_data['cat1']['searchResults']['listResults']
        print(2)
    except KeyError:
        jason_data = json_data
    return jason_data


def parse(zipcode):
    """Scrape data about houses from Zillow

    Args:
        zipcode (int): 5-digit zipcode to scrape

    Returns:
        Dictionary containing values for all variables for all listings
    """
    # Create initial url and get pages
    url, pages, parser = get_url_pages(zipcode)
    # print(url)
    # Initialize an empty list to store data
    data = []
    # Exclude missing zip codes
    if pages < 50:
        for i in range(1, pages+1):
            # The next section scrapes data from each page
            # First, we implement a while loop because Zillow is annoying and doesn't always format their pages consistently. We will loop over each url up to 10 times or until we get the properly formatted data
            x = 0
            while x < 1:
                # Then, try to extract the desired data. If the data is there, increment x to end the while loop
                try:
                    # Request the url
                    url, parser = get_url_pages(zipcode, firstPage=False, i=i)
                    print('Scraping page {}: {}'.format(i, url))
                    # Next, extract the raw json data
                    raw_json_data = parser.xpath(
                        '//script[@data-zrr-shared-data-key="mobileSearchPageStore"]//text()')
                    # Remove formatting strings
                    cleaned_data = clean(raw_json_data).replace('<!--', "").replace("-->", "")
                    # Convert data to json format
                    json_data = json.loads(cleaned_data)
                    jason_data = json_data['searchResults']['listResults']
                    x = 1
                    if i % 2 == 0:
                        sleep(5)
                # If the data's not there, try again and increment x toward the 10 try maximum.
                except AttributeError:
                    print('Trying again')
                    x += .1
                # If the data's not there, try again and increment x toward the 10 try maximum.
                except KeyError:
                    print('Trying again')
                    x += .1
            # For each result in the list
            try:
                for zipid in jason_data[:len(jason_data)]:
                    # Extract homedata
                    homedata = zipid['hdpData']['homeInfo']
                    # Append homedata to the master data holder
                    data.append(homedata)
            except KeyError:
                print('Failed to scrape Page {} in Zipcode {}'.format(i, zipcode))
                continue  # Skip this page
    return data


def scrapeCity(city, for_sale=True, save=True):
    # Assign an empty list to store zipcodes in the city
    zips = []
    for zip in zipcodes.filter_by(city=city):
        zips.append(zip['zip_code'])
    printText('There are {} zipcodes in {}'.format(len(zips), city))
    # Assign an empty list to store each listing
    data = []
    for zip in zips:
        try:
            # Extract zip data
            zip_data = parse(zip)
            # Append zip data to data
            data.append(zip_data)
            print('I\'m taking a quick break. Be right back')
            sleep(5)
        except IndexError:
            printText('No current listings in zipcode {}'.format(zip))
            continue  # skip zip code
    # Unpack list of list of dicts into a single list of dicts
    data = [inner for outer in data for inner in outer]
    if save:
        # Save data as a CSV File
        fileName = city.replace(' ', '_')
        pd_data = pd.DataFrame(data, index=None)
        pd_data.set_index('zpid', inplace=True)
        # Filter properties by type
        if for_sale:
            pd_data = pd_data[pd_data['homeStatusForHDP'] == 'FOR_SALE']
            pd_data.drop('homeStatusForHDP', axis=1, inplace=True)
        pd_data['url'] = home_url(pd_data)
        save_csv(pd_data, fileName)
    # Return list of dicts
    return data


headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5)'
           'AppleWebKit 537.36 (KHTML, like Gecko) Chrome',
           'Accept': 'text/html,application/xhtml+xml,application/xml;'
           'q=0.9,image/webp,*/*;q=0.8'}

data = scrapeCity('Los Angeles')
