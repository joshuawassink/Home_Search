home_search is a set of functions that can use listing level data to provide real estate market insights in real time. These functions are currently under development and may return errors.

CURRENT USEAGE
The primary script, zillowscrape.py, extracts listing level data from Zillow based on user-specified city or zip code preferences. This data is then fed to zillowCluster.py, which identifies and describes meaningful patterns in the data. zillowmodel.py (currently in development) uses machine learning algorithms to generate forward-looking insights and guide decision-making based on homes' characteristics, pricing, and location.

COMING SOON:
additional features (currently in development), will provide a layer of neighborhood characteristics on top of the existing functionalities. These characteristics, will be used to sort houses according to their local contexts, and user-preferences (e.g., commute-time). This sorting process will winnow the list of results reported by the web-scraper down to those in neighorhoods of greatest interest.