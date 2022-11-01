import argparse
import json

import pandas as pd
from tqdm import tqdm

from config.configs import exclusion_list
from utils.bias_analysis_utils import find_names

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/Yelp_cities')
    p = parser.parse_args()

    city_df = pd.read_csv('{}/{}_reviews.csv'.format(p.data_dir, p.city_name), lineterminator='\n')
    city_df.rename(columns={'stars': 'review_stars', 'text': 'review_text'}, inplace=True)
    sub_df = city_df[['review_text', 'price_lvl']].groupby('price_lvl')['review_text'].apply(' . '.join).reset_index()

    # get a dictionary to store people's names
    priceName_dict = dict()
    for idx, row in tqdm(sub_df.iterrows()):
        price = row['price_lvl']
        bus_reviews = row['review_text']
        people_names = find_names(bus_reviews, exclusion_list)
        if len(people_names) > 0:
            people_names_str = ' . '.join(people_names)
            priceName_dict[price] = priceName_dict.get(price, '') + ' . ' + people_names_str

    for price in priceName_dict.keys():
        with open('data/names/{}_peopleNames_{}priceLvl.json'.format(p.city_name, price), 'w') as fp:
            json.dump({price: priceName_dict[price]}, fp)
