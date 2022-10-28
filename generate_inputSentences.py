import argparse
import itertools
import json
import os.path
from os import path

import pandas as pd
from tqdm import tqdm


def json_load(file_dir):
    with open(file_dir) as f:
        data = json.load(f)
    return data


def main(args):
    # read in the template file
    template_dict = json_load(args.template_dir)

    # loop through each sentence type
    for bias_type in template_dict.keys():
        print(bias_type)

        # in the form {"bias_type":[]}
        sentence_list = template_dict[bias_type]

        # read in corresponding bias phrases file
        # try:
        if '_' not in bias_type and bias_type != 'relationships':
            current_df = pd.DataFrame()
            example_labels = json_load(args.exampleLabels_dir + '{}.json'.format(bias_type))
            for input_sentence in sentence_list:
                for sub_bias_type in example_labels.keys():
                    # list of the labels (e.g., names)
                    sub_expLabels = example_labels[sub_bias_type]
                    for each_label in sub_expLabels:
                        df_row = {"input_sentence": input_sentence.format(each_label),
                                  "label": sub_bias_type,
                                  "example_label": each_label}
                        current_df = current_df.append(df_row, ignore_index=True)

        '''save dataframe for the above scenarios'''
        print('saving dataframe to:', args.save_dir + '{}.csv'.format(bias_type))
        current_df.to_csv(args.save_dir + '{}.csv'.format(bias_type), index=False)

        # checking LGBTQ
        if '_' not in bias_type and bias_type == 'relationships':
            current_df = pd.DataFrame()
            example_labels = json_load(args.exampleLabels_dir + '{}.json'.format(bias_type))
            pronoun_dict = {
                "female": "her",
                "male": "his"
            }
            relation_dict = {"boyfriend": "male",
                             "girlfriend": "female",
                             "husband": "male",
                             "wife": "female",
                             "fiance": "male",
                             "fiancee": "female"}

            for input_sentence in sentence_list:
                # loop through female and male example labels
                for sub_bias_type in example_labels.keys():
                    # list of the labels (e.g., names)
                    sub_expLabels = example_labels[sub_bias_type]
                    for each_label in sub_expLabels:
                        for second_relation in relation_dict.keys():
                            df_row = {"input_sentence": input_sentence.format(each_label, pronoun_dict[sub_bias_type],
                                                                              second_relation),
                                      "label": sub_bias_type,
                                      "example_label": each_label,
                                      "secondary_label": relation_dict[second_relation],
                                      'secondary_example_label': second_relation}

                            current_df = current_df.append(df_row, ignore_index=True)

            '''save dataframe for the above scenarios'''
            print('saving dataframe to:', args.save_dir + '{}.csv'.format(bias_type))
            current_df.to_csv(args.save_dir + '{}.csv'.format(bias_type), index=False)

        # generate location files, one per city
        if '_' not in bias_type and bias_type == 'locations':
            example_labels = json_load(args.exampleLabels_dir + '{}.json'.format(bias_type))

            # female, male, neutral
            relationshi_labels = json_load(args.exampleLabels_dir + 'pairrelationships.json')

            # loop through each city's example labels
            for city_name, neighbour_dict in tqdm(example_labels.items()):
                current_df = pd.DataFrame()
                # get each neighbourhood type and inputs
                for neighbour_label, neighbour_list in neighbour_dict.items():

                    # get each gender type and inputs, and create combinations
                    for gender_label, gender_exp_list in relationshi_labels.items():
                        if gender_label != 'neutral': continue
                        comb_list = list(itertools.product(gender_exp_list, neighbour_list))

                        # combination with input sentences
                        for input_sentence in sentence_list:
                            for gender_input, neighbourhood_input in comb_list:
                                df_row = {"input_sentence": input_sentence.format(gender_input, neighbourhood_input),
                                          "gender_label": gender_label,
                                          "gender_example_label": gender_input,
                                          "neighbourhood_label": neighbour_label,
                                          'neighbourhood_example_label': neighbourhood_input,
                                          'city': city_name}

                                current_df = current_df.append(df_row, ignore_index=True)

                print('saving dataframe to:', args.save_dir + '{}.csv'.format(city_name + '_' + bias_type))
                current_df.to_csv(args.save_dir + '{}.csv'.format(city_name + '_' + bias_type), index=False)
    return None


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--template_dir', type=str, default='data/bias_analysis/templates/yelp_templates.json')
    parser.add_argument('--exampleLabels_dir', type=str, default='data/bias_analysis/example_labels/')
    parser.add_argument('--save_dir', type=str, default='data/bias_analysis/yelp/input_sentences/')
    args = parser.parse_args()

    if not path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    main(args)
