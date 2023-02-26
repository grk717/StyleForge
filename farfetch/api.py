import requests
import pandas as pd


class Api:
    def __init__(self):
        self.baseUrl = r'https://www.farfetch.com/plpslice/listing-api/products-facets'
        self.headers = {
            'Origin': 'https://www.farfetch.com'
            , 'Referer': 'https://www.farfetch.com/shopping/m/items.aspx'
            , 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'
                            ' Chrome/75.0.3770.100 Safari/537.36'
            }
        self.view = 180
        self.category = None
        self.pagetype = 'Shopping'
        self.gender = None
        self.pricetype = 'FullPrice'
        self.page = None
        self.params = '?view=%d&pagetype=%s&pricetype=%s&page=%d'
        self.product_list = None
        self.df = pd.DataFrame()

    def buildUrl(self):
        self.parameters = self.params % (
            self.view,
            self.pagetype,
            self.pricetype,
            self.page
        )
        return self.baseUrl + self.parameters

    def get_listings(self, page=1):
        self.page = page
        self.request = requests.get(self.buildUrl(), headers=self.headers)
        self.response = self.request.json()
        return self.response

    def parse_products(self, response):
        if response['listingItems'] is not None:
            self.product_list = self.response['listingItems']['items']
            if len(self.df) == 0:
                self.df = pd.json_normalize(self.product_list)
            else:
                self.df = pd.concat([self.df, pd.json_normalize(self.product_list)], ignore_index=True)

