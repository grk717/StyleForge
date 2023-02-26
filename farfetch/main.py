import api
import datetime

date = datetime.datetime.today().strftime('%m-%d-%y')

api = api.Api()
total_pages = api.get_listings()['listingPagination']['totalPages']


def run():
    for page in range(1, 5):
        print(page)
        api.parse_products(
            api.get_listings(page=page)
        )
    return api.df


run()
api.df.to_csv('current_farfetch_listings' + date + '.csv')

