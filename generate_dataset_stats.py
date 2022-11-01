# -*- coding: utf-8 -*-
"""
Automatically generated by Colaboratory.
Original file is located at
    https://colab.research.google.com/drive/1-hiZ1aPYsB8Ye22AcGnOrkKq1wnUTYI6
"""

import argparse
import gender_guesser.detector as gender
import json
import numpy as np
import pandas as pd
import seaborn as sns
from ethnicolr import census_ln
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm

from config.configs import city_list, exclusion_list
from utils.utils import get_latex_table_from_stats_df, pickle_dump, pickle_load

nltk.download('stopwords')
stopwords = stopwords.words('english')
sns.set_style("darkgrid")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/Yelp_cities')
    parser.add_argument('--relationships_label_dir', type=str, default='data/bias_analysis/example_labels/relationships.json')
    parser.add_argument('--pairrelationships_label_dir', type=str, default='data/bias_analysis/example_labels/pairrelationships.json')
    parser.add_argument('--gender_stats_dir', type=str, default='data/bias_analysis/yelp/gender_statistics.csv')
    parser.add_argument('--get_gender_relationship_stats', action='store_true', help='whether to scan the review text and generate the gender stats')
    p = parser.parse_args()

    """
    1.Check Gender
    """
    print('-' * 20 + '1. Check Gender' + '-' * 20)
    statistics_df = pd.DataFrame(columns=['city', 'price_lvl', 'bias', 'label', 'percent_stats'])
    city_name_list = city_list.copy()
    city_name_list.sort()

    print('1.1 Get gender stats using names')
    d = gender.Detector()
    for city_name in city_name_list:
        for set_price in range(1, 5):
            with open('data/debias/{}_peopleNames_{}priceLvl.json'.format(city_name, set_price), 'r') as fp:
                all_names = fp.read()
                input_text = [txt for txt in all_names.split(' . ') if txt.lower() not in exclusion_list]
                print('-----------------------------------------------------------------')
                gender_result = {}
                for i in input_text:
                    gender_name = d.get_gender(i)
                    gender_result[gender_name] = gender_result.get(gender_name, 0) + 1

                print(city_name + ' ' + '$' * set_price)
                print(gender_result)

                # not counting the "mostly_" tags
                male_total = int(gender_result.get('male'))
                female_total = int(gender_result.get('female'))

                known_total = male_total + female_total
                male_percent = 100 * male_total / known_total
                female_percent = 100 * female_total / known_total

                print("male names percent = " + str(male_percent) + "%")
                print("female names percent = " + str(female_percent) + "%")

                for bias in ['male', 'female']:
                    row = {
                        'city': city_name,
                        'price_lvl': '$' * set_price,
                        'bias': 'gender',
                        'label': bias,
                        'percent_stats': male_percent if bias == 'male' else female_percent
                    }
                    # statistics_df = statistics_df.append(row, ignore_index=True)
                    statistics_df = pd.concat([statistics_df, pd.DataFrame.from_records([row])])

    """Check by using relationship words"""
    print('1.2 Get gender stats using relationship words')

    if p.get_gender_relationship_stats:
        with open(p.relationships_label_dir) as json_file:
            gender_dict = json.load(json_file)
        with open(p.pairrelationships_label_dir) as json_file:
            gender_dict1 = json.load(json_file)
        for key in gender_dict.keys():
            gender_dict[key] = list(set(gender_dict[key] + gender_dict1[key]))

        # scan review text
        gender_stats_dict = {city: {price: {label: 0 for label in gender_dict.keys()} for price in range(1, 5)} for city in city_list}
        for city_name in city_list:
            print('---------{}---------'.format(city_name))
            city_df = pd.read_csv('{}/{}_reviews.csv'.format(p.data_dir, city_name), lineterminator='\n')
            city_df.rename(columns={'text': 'review_text'}, inplace=True)

            load_dir = 'data/Yelp_cities/{}_trainValidTest_5f/'.format(city_name)
            labels = pickle_load(load_dir + 'labels.pickle'.format(city_name))
            city_df = city_df[city_df['business_id'].isin(labels)]
            if city_name == 'Toronto':
                city_df['price'] = city_df['price'].apply(lambda x: int(len(str(x))))
            else:
                city_df['price'] = city_df['price'].apply(lambda x: int(float(x)))
                print('Number of reviews being used for city {}: {} '.format(city_name, len(city_df)))

            data = city_df[['price', 'review_text']].values.tolist()

            with tqdm(total=len(data)) as pbar:
                for idx, v in enumerate(data):
                    pbar.update(1)
                    price_lvl = v[0]
                    review_text = v[1]

                    for key in gender_dict.keys():
                        for value in gender_dict[key]:
                            temp_count = review_text.lower().count(value)
                            if temp_count >= 1:
                                gender_stats_dict[city_name][price_lvl][key] += temp_count

                    if idx >= 100000 and idx % 100000 == 0:
                        print('--------------{}-------------'.format(idx))
                        print(gender_stats_dict[city_name])
                        pickle_dump(p.gender_stats_dir.replace('.csv', '.pickle'), gender_stats_dict)

    gender_stats_dict = pickle_load(p.gender_stats_dir.replace('.csv', '.pickle'))
    # calculate percentages and flatten the dictionary
    stats_dict = {'city': list(),
                  'price_lvl': list(),
                  'bias': list(),
                  'label': list(),
                  'percent_stats': list()}

    temp_dict = None
    for city_name in gender_stats_dict.keys():
        for price_lvl in gender_stats_dict[city_name]:
            temp_dict = gender_stats_dict[city_name][price_lvl]
            female_count = temp_dict['female']
            male_count = temp_dict['male']
            temp_dict['female'] = 100 * female_count / (female_count + male_count)
            temp_dict['male'] = 100 * male_count / (female_count + male_count)

            for label, pct in temp_dict.items():
                stats_dict['city'].append(city_name)
                stats_dict['price_lvl'].append('$' * price_lvl)
                stats_dict['bias'].append('gender')
                stats_dict['label'].append(label)
                stats_dict['percent_stats'].append(pct)

    stats_df = pd.DataFrame.from_dict(stats_dict)
    stats_df.to_csv(p.gender_stats_dir, index=False)
    get_latex_table_from_stats_df(stats_df, p.gender_stats_dir.replace('.csv', '.txt'))

    """
    2.Check race
    """
    print('-' * 20 + '2. Check Race' + '-' * 20)
    for city_name in city_name_list:
        for set_price in range(1, 5):
            with open('data/debias/{}_peopleNames_{}priceLvl.json'.format(city_name, set_price), 'r') as fp:
                all_names = json.loads(fp.read())[str(set_price)]
                input_text = [txt for txt in all_names.split(' . ') if txt.lower() not in exclusion_list]
                input_text = input_text[1:]

                df_names = pd.DataFrame(input_text, columns=['name'])
                df_result = census_ln(df_names, 'name')
                df_result = df_result.fillna(0)
                df_result = df_result.replace('(S)', 0)

                # convert to numerical values
                for col in ['pctwhite', 'pctblack', 'pctapi', 'pctaian', 'pct2prace', 'pcthispanic']:
                    df_result[col] = pd.to_numeric(df_result[col])

                    df_result_val = df_result[['pctwhite', 'pctblack', 'pctapi', 'pctaian', 'pct2prace', 'pcthispanic']]
                    df_result_val['max'] = df_result_val.astype('float64').idxmax(axis=1)

                    # set unknown
                    df_result_val['max'] = np.where(((df_result_val['pctwhite'] == 0)
                                                     & (df_result_val['pctblack'] == 0)
                                                     & (df_result_val['pctapi'] == 0)
                                                     & (df_result_val['pctaian'] == 0)
                                                     & (df_result_val['pct2prace'] == 0)
                                                     & (df_result_val['pcthispanic'] == 0)), 'unknown', df_result_val['max'])

                race_result = {}
                for i, j in df_result_val[['max']].iterrows():
                    race_result[j[0]] = race_result.get(j[0], 0) + 1
                    print('-----------------------------------------------------------------')
                    print(city_name + ' ' + '$' * set_price)
                    print(race_result)
                    if 'pctwhite' in race_result.keys():
                        white_total = int(race_result.get('pctwhite'))
                    else:
                        white_total = 0

                    black_total = 0
                    for key in race_result.keys():
                        if key != 'pctwhite' and key != 'unknown':
                            black_total += int(race_result.get(key))

                    known_total = white_total + black_total
                    white_percent = 100 * white_total / (known_total + 1e-4)
                    black_percent = 100 * black_total / (known_total + 1e-4)
                    print("white names percent = " + str(white_percent) + "%")
                    print("black names percent = " + str(black_percent) + "%")

                for bias in ['white', 'black']:
                    row = {
                        'city': city_name,
                        'price_lvl': '$' * set_price,
                        'bias': 'race',
                        'label': bias,
                        'percent_stats': white_percent if bias == 'white' else black_percent
                    }
                    # statistics_df = statistics_df.append(row, ignore_index=True)
                    statistics_df = pd.concat([statistics_df, pd.DataFrame.from_records([row])])

    # This file contains the entire statistics
    statistics_df.to_csv('bias_analysis/yelp/statistics.csv', index=False)
    get_latex_table_from_stats_df(statistics_df, 'bias_analysis/yelp/statistics.txt')
