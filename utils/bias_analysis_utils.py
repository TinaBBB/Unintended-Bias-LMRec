import numpy as np
import pandas as pd
import seaborn as sns

from config.configs import confidence_interval_dict
from sklearn.utils import shuffle
from utils.utils import concat_city_df
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype
from scipy import stats


def get_deviation_score(deviation_list, deviation_score_list, deviation_score_dict, decimal_places=4):
    abs_deviation = abs(deviation_list)
    squared_deviation = (deviation_list.reshape(1, -1) @ deviation_list.reshape(-1, 1)).flatten()[0]

    for deviation_score in deviation_score_list:
        if deviation_score == 'Mean AD':
            score = round(np.mean(abs_deviation), decimal_places)
        elif deviation_score == 'Median AD':
            score = round(np.median(abs_deviation), decimal_places)
        elif deviation_score == 'Max AD':
            score = round(max(abs_deviation), decimal_places)
        elif deviation_score == 'Std':
            score = round(np.sqrt(squared_deviation / len(deviation_list)), decimal_places)
        else:
            assert False, 'Unexpected deviation score'
        deviation_score_dict[deviation_score] = score
    return None


def get_counts(df_sum, jointCount_dict, cat_name, label):
    joint_count = jointCount_dict[cat_name].get(label, None)
    if not joint_count:
        x_label = 0
        x_label_count = 0
    else:
        x_label = joint_count / (df_sum[df_sum['label'] == label]['counts'].iloc[0])  # prob
        x_label_count = joint_count
    return x_label, x_label_count


def get_counts_wFold(df_, jointCount_dict, cat_name, label, fold_num):
    try:
        joint_count = jointCount_dict[cat_name][fold_num].get(label, None)
    except:
        joint_count = None

    if not joint_count:  # if empty
        x_label = 0
        x_label_count = 0
        x_count = 0
    else:
        x_count = df_[(df_['label'] == label) & (df_['fold'] == fold_num)]['counts'].iloc[0]
        x_label = joint_count / x_count  # prob
        x_label_count = joint_count
    return x_label, x_label_count, x_count


def get_counts_so(df_sum_sex_orient_all, jointCount_dict, cat_name, label):
    joint_count = jointCount_dict[cat_name].get(label, None)
    if not joint_count:  # if empty
        x_label = 0
        x_label_count = 0
    else:
        x_label = joint_count / (df_sum_sex_orient_all[df_sum_sex_orient_all['label'] == label.split('_')[0]]['counts'].iloc[0])  # prob
        x_label_count = joint_count

    return x_label, x_label_count


def get_counts_so_wFolds(df_, jointCount_dict, cat_name, label, fold_num):
    try:
        joint_count = jointCount_dict[cat_name][fold_num].get(label, None)
    except:
        joint_count = None

    if not joint_count:
        x_label = 0
        x_label_count = 0
        x_count = 0
    else:
        x_count = df_[(df_['label'] == label.split('_')[0]) & (df_['fold'] == fold_num)]['counts'].iloc[0]
        x_label = joint_count / x_count  # prob
        x_label_count = joint_count

    return x_label, x_label_count, x_count


def get_fraction_df(bias_ratio_df, bias_polarity, calculation_method='fraction', minmax_normalization=False):
    def calculate(idx, row):
        if calculation_method == 'fraction':
            temp_rec_frac = temp_df1.ratio_recommendation.values[idx] / (temp_df2.ratio_recommendation.values[idx] + 1e-4)
            temp_stats_frac = temp_df1.percent_stats.values[idx] / (temp_df2.percent_stats.values[idx] + 1e-4)
        elif calculation_method == 'difference':
            temp_rec_frac = temp_df1.ratio_recommendation.values[idx] - temp_df2.ratio_recommendation.values[idx]
            temp_stats_frac = temp_df1.percent_stats.values[idx] - temp_df2.percent_stats.values[idx]
        else:
            assert False, 'Unrecognized calculation method'

        row['label'] = '/'.join(bias_polarity)
        row['ratio_recommendation'] = temp_rec_frac
        row['percent_stats'] = temp_stats_frac

        return row

    # For the same city & price_lvl, get male/female ratio
    bias_frac_df = pd.DataFrame(columns=bias_ratio_df.columns)

    for city_name in bias_ratio_df.city.unique():
        for price_level in bias_ratio_df.price_lvl.unique():
            temp_df1 = bias_ratio_df[
                (bias_ratio_df['city'] == city_name) & (bias_ratio_df['price_lvl'] == price_level) & (bias_ratio_df['label'] == bias_polarity[0])]
            temp_df2 = bias_ratio_df[
                (bias_ratio_df['city'] == city_name) & (bias_ratio_df['price_lvl'] == price_level) & (bias_ratio_df['label'] == bias_polarity[1])]
            try:
                row = temp_df1.to_dict('records')[0].copy()
            except:
                print(temp_df1.to_dict('records'))
                print(city_name, price_level)

            if 'fold_num' in bias_ratio_df.columns:
                for idx in range(bias_ratio_df.fold_num.nunique()):
                    row = calculate(idx, row)
                    # bias_frac_df = bias_frac_df.append(row, ignore_index=True)
                    bias_frac_df = pd.concat([bias_frac_df, pd.DataFrame.from_records([row])])
            else:
                row = calculate(0, row)
                # bias_frac_df = bias_frac_df.append(row, ignore_index=True)
                bias_frac_df = pd.concat([bias_frac_df, pd.DataFrame.from_records([row])])

    if minmax_normalization:
        ratio_recommendation = bias_frac_df['ratio_recommendation']
        percent_stats = bias_frac_df['percent_stats']

        bias_frac_df['ratio_recommendation'] = (ratio_recommendation - ratio_recommendation.min()) / (
                ratio_recommendation.max() - ratio_recommendation.min())
        bias_frac_df['percent_stats'] = (percent_stats - percent_stats.min()) / (percent_stats.max() - percent_stats.min())

    return bias_frac_df


# Calculate the ratio per price level numbers
def get_price_ratio_df(df_names):
    df_price_ratio = df_names.copy()
    df_price_ratio = df_price_ratio[['label', 'price_lvl', 'rank']]
    df_price_ratio_groupby = df_price_ratio.groupby(by=['label', 'price_lvl']).size().reset_index(name='counts')
    df_price_ratio_groupby = df_price_ratio_groupby.drop(df_price_ratio_groupby[df_price_ratio_groupby['label'] == 'neutral'].index)
    df_price_ratio_groupby['bias'] = df_price_ratio_groupby['label'].apply(lambda x: 'gender' if x in ['female', 'male'] else 'race')

    df_price_ratio_plot = pd.DataFrame()
    for idx, row in df_price_ratio_groupby.iterrows():
        price = row['price_lvl']
        bias = row['bias']
        deno = sum(df_price_ratio_groupby[(df_price_ratio_groupby['price_lvl'] == price) & (df_price_ratio_groupby['bias'] == bias)].counts)
        ratio = row['counts'] / deno
        row['ratio'] = ratio
        # df_price_ratio_plot = df_price_ratio_plot.append(row)
        df_price_ratio_plot = pd.concat([df_price_ratio_plot, pd.DataFrame.from_records([row])])

    return df_price_ratio_plot


# With the price ratio dataframe with fold numbers
def get_price_ratio_df_wFold(df_names, fold=10):
    df_price_ratio = df_names.copy()
    df_price_ratio = df_price_ratio[['label', 'price_lvl', 'rank']]
    df_price_ratio = shuffle(df_price_ratio, random_state=0)
    index_list = np.linspace(0, len(df_price_ratio), fold + 1, dtype=int)

    df_price_ratio_plot = pd.DataFrame()
    for fold_num, index_position in enumerate(index_list):
        if fold_num == fold: break
        df_price_ratio_groupby = df_price_ratio[index_list[fold_num]:index_list[fold_num + 1]].groupby(
            by=['label', 'price_lvl']).size().reset_index(
            name='counts')
        df_price_ratio_groupby = df_price_ratio_groupby.drop(df_price_ratio_groupby[df_price_ratio_groupby['label'] == 'neutral'].index)
        df_price_ratio_groupby['bias'] = df_price_ratio_groupby['label'].apply(lambda x: 'gender' if x in ['female', 'male'] else 'race')

        for idx, row in df_price_ratio_groupby.iterrows():
            price = row['price_lvl']
            bias = row['bias']
            deno = sum(df_price_ratio_groupby[(df_price_ratio_groupby['price_lvl'] == price) & (df_price_ratio_groupby['bias'] == bias)].counts)
            ratio = row['counts'] / deno
            row['ratio'] = ratio
            row['fold_num'] = fold_num
            # df_price_ratio_plot = df_price_ratio_plot.append(row)
            df_price_ratio_plot = pd.concat([df_price_ratio_plot, pd.DataFrame.from_records([row])])

    return df_price_ratio_plot


def getOccupation_df(city_list, bias_placeholder_dir, parsed_args, use_folds, fold_number=10):
    df_occupation = concat_city_df(city_list, bias_placeholder_dir, parsed_args, 'yelp_qa_occupations.csv')
    df_add = concat_city_df(city_list, bias_placeholder_dir, parsed_args, 'yelp_qa_additionalOccupations.csv')
    df_occupation = df_occupation.append(df_add)

    # remove farm and institution
    df_occupation = df_occupation[df_occupation['example_label'].isin(['farm', 'institution', 'bar', 'chemical lab']) == False]
    df_occupation['example_label'] = np.where(df_occupation['example_label'] == 'dancing studio', 'dance studio', df_occupation['example_label'])
    df_occupation['type'] = np.where(df_occupation['example_label'].isin(['church', 'mosque', 'synagogue']), 'religion', 'location')

    df_occupation_groupby_avg = df_occupation.groupby(['example_label', 'type'])['price_lvl'].mean().reset_index(name='average_price_lvl')
    df_occupation_groupby_avg = df_occupation_groupby_avg.sort_values(by='average_price_lvl')
    df_occupation_groupby_avg['city'] = 'Average'

    # get the averaged price value for each city
    df_occupation_groupby_avg_all = df_occupation.groupby(['city', 'example_label', 'type'])['price_lvl'].mean().reset_index(
        name='average_price_lvl')
    df_occupation_groupby_avg_all = df_occupation_groupby_avg_all.sort_values(by='average_price_lvl')

    if use_folds:
        # get confidence interval by folds
        np.random.seed(68)
        df_occupation_folded = df_occupation[['example_label', 'type', 'price_lvl']]
        df_occupation_folded['fold'] = np.random.randint(0, fold_number, df_occupation_folded.shape[0])
        # get the averaged price value for each city
        df_occupation_groupby_avg_all = df_occupation_folded.groupby(['fold', 'example_label', 'type'])['price_lvl'].mean().reset_index(
            name='average_price_lvl')
        df_occupation_groupby_avg_all = df_occupation_groupby_avg_all.sort_values(by='average_price_lvl')

    cat_example_order = CategoricalDtype(
        df_occupation_groupby_avg['example_label'].tolist(),
        ordered=True
    )

    df_occupation_groupby_avg_all['example_label'] = df_occupation_groupby_avg_all['example_label'].astype(cat_example_order)

    return df_occupation_groupby_avg_all


def create_occupationPlotDf(df_occupation_groupby_avg_all, type_to_show):
    df_temp = df_occupation_groupby_avg_all[df_occupation_groupby_avg_all['type'] == type_to_show]

    df_plot = pd.DataFrame(columns=['Example Label', 'Average Price Level', 'Error'])
    for lbl in df_temp['example_label'].unique():
        data = list(df_temp[df_temp['example_label'] == lbl]['average_price_lvl'])
        mean, err = mean_confidence_interval(data, 0.90, len(data))
        # df_plot = df_plot.append({'Example Label': lbl, 'Average Price Level': mean, 'Error': err}, ignore_index=True)
        df_plot = pd.concat([df_plot, pd.DataFrame.from_records([{'Example Label': lbl, 'Average Price Level': mean, 'Error': err}])])

    df_plot = df_plot.sort_values(by='Average Price Level', ascending=True)
    return df_plot


def mean_confidence_interval(data, confidence=0.95, n=None):
    conf_dict = confidence_interval_dict
    return np.average(data), conf_dict[confidence] * np.std(data) / np.sqrt(n)


def plot_barChart(data, x, y, h, lb, ub, a_axis, title, bar_width, fontsize, y_min, neutralize=False):
    errLo = data.pivot(index=x, columns=h, values=lb)
    errHi = data.pivot(index=x, columns=h, values=ub)
    err = []
    for col in errLo:
        err.append([errLo[col].values, errHi[col].values])
    err = np.abs(err)
    p = data.pivot(index=x, columns=h, values=y)
    ax = p.plot(kind='bar', yerr=err, ax=a_axis, width=bar_width,
                xlabel='', ylabel='')
    ax.set_title(title, fontdict={'fontsize': fontsize})
    ax.set_ylim(y_min)

    if neutralize:
        ax.axhline(0.5, color="gray", linestyle="--", label='neutral reference')

    plt.legend(loc='upper center', bbox_to_anchor=[0.5, 0.99], ncol=4,
               bbox_transform=plt.gcf().transFigure, fontsize=13)


def plot_scatter(df, bias_polarity, calculation_method, x_data="percent_stats", y_data="ratio_recommendation", title=None, withTrend=False,
                 save_dir='bias_analysis/yelp/figures_5f/correlation.pdf'):
    if withTrend:
        ax = sns.lmplot(x=x_data, y=y_data, data=df.astype('float64'), height=5, aspect=1.1)
    else:
        ax = sns.scatterplot(data=df[[x_data, y_data]], x=x_data, y=y_data, height=5, aspect=1.1)

    assert calculation_method in ['fraction', 'difference'], 'Unrecognized calculation method'

    xLabel = 'Dataset statistics \n ({}{}{})'.format(bias_polarity[0],
                                                     '-' if calculation_method == 'difference' else '/',
                                                     bias_polarity[1])
    yLabel = 'Ratio in recommendation \n ({}{}{})'.format(bias_polarity[0],
                                                          '-' if calculation_method == 'difference' else '/',
                                                          bias_polarity[1])

    ax.set(xlabel=xLabel, ylabel=yLabel)
    cor, pvalue = stats.pearsonr(df[x_data], df[y_data])

    if title:
        plt.title(title + ' \n ({},{})'.format(round(cor, 3), round(pvalue, 3)))
    plt.savefig(save_dir, bbox_inches='tight')
    return cor, pvalue