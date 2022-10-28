import re
import pandas as pd
import pickle
import numpy as np
import tensorflow as tf
from tokenizers.implementations import BertWordPieceTokenizer
from tensorflow import keras
from tensorflow.keras import layers
from transformers import TFBertModel


def pickle_load(path):
    with open(path, 'rb') as fp:
        file = pickle.load(fp)
    return file


def pickle_dump(path, file):
    with open(path, 'wb') as f:
        pickle.dump(file, f)


def get_latex_table_from_stats_df(df, save_dir):
    print_df = df.copy()
    print_df['bias'] = print_df['bias'].apply(lambda x: x.capitalize())
    # print_df['price_lvl'] = print_df['price_lvl'].apply(lambda x: '$' * x)
    print_df['label'] = print_df['label'].apply(lambda x: x.capitalize() + '%')
    print_df['percent_stats'] = print_df['percent_stats'].apply(lambda x: round(100 * x, 2))

    print_df = print_df.pivot(index=['bias', 'price_lvl', 'label'], columns=['city'], values=['percent_stats'])
    print_df.columns = print_df.columns.to_series().str.join('_')
    print_df.columns = [name.replace('percent_stats_', '') for name in print_df.columns]
    print_df.index = print_df.index.set_names(names=['', '', ''])
    with open(save_dir, 'w') as tf:
        tf.write(print_df.style.to_latex())


def concat_city_df(city_list, bias_placeholder_dir, parsed_args, file_name):
    df_list_total = []
    for cityName in city_list:
        temp_bias_dir = bias_placeholder_dir.format(cityName, parsed_args.experiment)
        temp_url = temp_bias_dir + file_name
        temp_df_names = pd.read_csv(temp_url, index_col=0)
        if cityName == 'Toronto':
            temp_df_names['price_lvl'] = temp_df_names['price'].str.len()
        else:
            temp_df_names['price_lvl'] = temp_df_names['price']
        temp_df_names['city'] = cityName
        df_list_total.append(temp_df_names)
    df_city_all_plot = pd.concat(df_list_total, ignore_index=True)
    return df_city_all_plot


def split_cat(old_df):
    '''
      splits out the categories from a result dataframe and retur a new dataframe
      for yelp
    '''

    # create a new dataframe that splits the categories
    old_df.head()
    cat_replacements = {'&': ',',
                        '/': ',',
                        '(': ',',
                        ')': ''}

    new_df = pd.DataFrame(columns=old_df.columns)

    for index, row in old_df.iterrows():
        cats_string = row['categories']
        text = multireplace(cats_string, cat_replacements)
        cats_list = text.lower().split(',')
        for cat in cats_list:
            row_copy = row.copy()
            row_copy['categories'] = cat.strip()
            new_df = new_df.append(row_copy)

    return new_df


def multireplace(string, replacements):
    """
    Given a string and a replacement map, it returns the replaced string.
    :param str string: string to execute replacements on
    :param dict replacements: replacement dictionary {value to find: value to replace}
    :rtype: str

    """
    # Place longer ones first to keep shorter substrings from matching where the longer ones should take place
    # For instance given the replacements {'ab': 'AB', 'abc': 'ABC'} against  the string 'hey abc', it should produce 'hey ABC' and not 'hey ABc'
    substrs = sorted(replacements, key=len, reverse=True)

    # Create a big OR regex that matches any of the substrings to replace
    regexp = re.compile('|'.join(map(re.escape, substrs)))

    # For each match, look up the new string in the replacements
    return regexp.sub(lambda match: replacements[match.group(0)], string)


# convert input text phrase to tokens and attention mask
def get_input_list(review_list, max_len):
    tokenizer = BertWordPieceTokenizer("bert_base_uncased/vocab.txt", lowercase=True)
    input_id_list = list()
    attention_mask_list = list()
    for review in review_list:
        # Process text
        input_ids = tokenizer.encode(review).ids
        attention_mask = [1] * len(input_ids)
        padding_length = max_len - len(input_ids)
        if padding_length > 0:  # pad
            input_ids = input_ids + ([0] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
        else:
            input_ids = input_ids[0:max_len]
            input_ids[-1] = 102  # separation token
            attention_mask = attention_mask[0:max_len]

        input_id_list.append(input_ids)
        attention_mask_list.append(attention_mask)
    return [np.array(input_id_list), np.array(attention_mask_list)]


# model
def create_model(max_len, labels, learning_rate=5e-5, ):
    encoder = TFBertModel.from_pretrained("bert-base-uncased")
    input_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32)

    embedding = encoder(
        input_ids, attention_mask=attention_mask
    )['pooler_output']

    dense = layers.Dense(1024, activation='relu')(embedding)
    out = layers.Dense(len(labels), activation='softmax')(dense)

    model = keras.Model(
        inputs=[input_ids, attention_mask],
        outputs=out, )

    loss = keras.losses.SparseCategoricalCrossentropy()
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model
