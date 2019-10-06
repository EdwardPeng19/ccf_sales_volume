import pandas as pd
import numpy as np
import pickle
import sys
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
P_DATA = "./data/"
P_SUBMIT = "./submit/"
def data_describe(df, names):
    print(f'data-size:{df.shape}')
    print(f'data-columns:{df.columns}')
    print('主要字段分布情况')
    for n in names:
        df_ncounts = df[n].value_counts()
        print(f'n:{n}, 共有{df_ncounts.shape}个值')
        print(df_ncounts)

def train_input():
    train_sales_model = pd.read_csv(P_DATA + 'train_sales_data.csv', encoding='utf-8')
    train_search = pd.read_csv(P_DATA + 'train_search_data.csv', encoding='utf-8')
    train_user_reply = pd.read_csv(P_DATA + 'train_user_reply_data.csv', encoding='utf-8')
    evaluation_public = pd.read_csv(P_DATA + 'evaluation_public.csv', encoding='utf-8')
    workdays_info = pd.read_csv(P_DATA + 'workdays.csv', encoding='utf-8')

    train_sales_model['bt_ry_mean'] = train_sales_model.groupby(['bodyType', 'regYear'])['salesVolume'].transform('mean')
    train_sales_model['ad_ry_mean'] = train_sales_model.groupby(['adcode', 'regYear'])['salesVolume'].transform('mean')
    train_sales_model['md_ry_mean'] = train_sales_model.groupby(['model', 'regYear'])['salesVolume'].transform('mean')

    data = pd.concat([train_sales_model, evaluation_public], ignore_index=True)
    data = data.merge(train_search, how='left', on=['province', 'adcode', 'model', 'regYear', 'regMonth'])
    data = data.merge(train_user_reply, how='left', on=['model', 'regYear', 'regMonth'])
    data = data.merge(workdays_info, how='left', on=['regYear', 'regMonth'])

    data['id'] = data['id'].fillna(0).astype(int)
    data['label'] = data['salesVolume']
    del data['salesVolume'], data['forecastVolum']
    #print(data.info())
    data['bodyType'] = data['model'].map(train_sales_model.drop_duplicates('model').set_index('model')['bodyType'])

    # 填充测试集特征值
    for col in ['carCommentVolum', 'newsReplyVolum', 'popularity', 'bt_ry_mean', 'ad_ry_mean', 'md_ry_mean']:
        lgb_col_na = pd.isnull(data[col])
        data[col] = data[col].replace(0, 1)
        data.loc[lgb_col_na, col] = \
            ((((data.loc[(data['regYear'].isin([2017])) & (data['regMonth'].isin([1, 2, 3, 4])), col].values /
                data.loc[(data['regYear'].isin([2016])) & (data['regMonth'].isin([1, 2, 3, 4])), col].values))) *
             data.loc[(data['regYear'].isin([2017])) & (data['regMonth'].isin([1, 2, 3, 4])), col].values * 1.03).round()
    return data

def cut_season(x):
    x = int(x)
    if x<=3:
        return 1
    if x<=6:
        return 2
    if x<=9:
        return 3
    if x>9:
        return 4

def window_rollth(x):
    return list(x)[0]


def feature_main(sales_path=None, offline=False):
    t = time.time()
    if sales_path:
        data = pd.read_pickle(P_DATA + 'train_pre.pk')
        if offline:
            sales_data = pd.read_csv(sales_path)[['y_hat','model_adcode_ym']]
            data = data.merge(sales_data, how='left', on='model_adcode_ym')
            #print(data[data['ym']==21][['model_adcode_ym','label','y_hat']])
            data['label'] = data.apply(lambda x: x['y_hat'] if x['y_hat']>0 else x['label'] , axis=1)
            #print(data[data['ym'] == 21][['model_adcode_ym', 'label', 'y_hat']])
        else:
            sales_data = pd.read_csv(sales_path).rename(columns={'forecastVolum':'salesNum'})
            data = data.merge(sales_data, how='left', on='id')
            data['label'] = data.apply(lambda x: x['salesNum'] if x['salesNum']>0 else x['label'] , axis=1)
            #data['label'] = data['label'].map(sales_data['forecastVolum'])
    else :
        data = train_input()
        # 窗口平移 #'popularity', 'carCommentVolum', 'newsReplyVolum'
        data['season'] = data.apply(lambda x: cut_season(x['regMonth']), axis=1)

        data['label'] = data['label'].fillna(0)
        data['popularity'] = data['popularity'].fillna(0)
        data['carCommentVolum'] = data['carCommentVolum'].fillna(0)
        data['newsReplyVolum'] = data['newsReplyVolum'].fillna(0)

        for i in ['bodyType', 'model']:
            data[i] = data[i].map(dict(zip(data[i].unique(), range(data[i].nunique()))))  # model:60 body:4

        data['ym'] = (data['regYear'] - 2016) * 12 + data['regMonth']  # 1-24 25 26 27 28
        data['model_adcode'] = data['adcode'] + data['model']  # 56：省份， 12：车型
        data['model_ym'] = data['model'] * 100 + data['ym']  # 34：车型， 12：年月
        data['adcode_ym'] = data['adcode'] + data['ym']  # 34：省份， 12：年月
        data['body_ym'] = data['bodyType'] * 100 + data['ym']  # 34：车身， 12：年月
        data['model_adcode_ym'] = data['model_adcode'] * 100 + data['ym']  # 78:省份 34：车型 ， 12：年月

        a = 6
        b = 4
        data['weightMonth'] = data['regMonth'].map({1: a, 2: a, 3: a, 4: a,
                                                    5: b, 6: b, 7: b, 8: b, 9: b, 10: b, 11: b, 12: b, })
        data['happyNY'] = 0  # 重要特征
        data.loc[(data['regYear'].isin([2016, 2018]) & data['regMonth'].isin([2])), 'happyNY'] = 1
        data.loc[(data['regYear'].isin([2017]) & data['regMonth'].isin([1])), 'happyNY'] = 1

        # data.loc[(data['regYear'].isin([2016]) & data['regMonth'].isin([1])), 'happyNY'] = 2
        # data.loc[(data['regYear'].isin([2016]) & data['regMonth'].isin([12])), 'happyNY'] = 2
        # data.loc[(data['regYear'].isin([2018]) & data['regMonth'].isin([1])), 'happyNY'] = 2
        #
        # data.loc[(data['regYear'].isin([2016]) & data['regMonth'].isin([3])), 'happyNY'] = 3
        # data.loc[(data['regYear'].isin([2017]) & data['regMonth'].isin([2])), 'happyNY'] = 3
        # data.loc[(data['regYear'].isin([2018]) & data['regMonth'].isin([3])), 'happyNY'] = 3

        # data_ = data[data['ym']<=24][['model','province','ym', 'popularity', 'carCommentVolum', 'newsReplyVolum','label', ]].set_index(['model','province','ym']).unstack()
        # from sklearn.cluster import KMeans
        # kmeans = KMeans(n_clusters=5)
        # kmeans.fit(data_.values)
        # data_['cluster_sale'] = kmeans.labels_
        # data_ = data_['cluster_sale'].reset_index()
        # print(data_['cluster_sale'].value_counts())
        # print(data.info())
        # data = data.merge(data_, on=['model','province'], how='left')
        # print(data.info())
        with open(P_DATA + 'train_pre.pk', 'wb') as train_f:
            pickle.dump(data, train_f)

    #print('shift 特征')
    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,13]:
        # 对销量和搜索量
        data['model_adcode_ym_{0}'.format(i)] = data['model_adcode_ym'] + i
        data_last = data[~data.label.isnull()].set_index('model_adcode_ym_{0}'.format(i))
        data[f'sales_{i}thMonth'] = data['model_adcode_ym'].map(data_last['label'])
        data[f'popularity_{i}thMonth'] = data['model_adcode_ym'].map(data_last['popularity'])
        #按省分组统计
        data['model_ym_{0}'.format(i)] = data['model_ym'] + i
        data_last = data[~data.label.isnull()].set_index('model_ym_{0}'.format(i))
        data_last1 = data_last[~data_last.index.duplicated()]
        data[f'comment_{i}thMonth'] = data['model_ym'].map(data_last1['carCommentVolum']) #评论和回复
        data[f'reply_{i}thMonth'] = data['model_ym'].map(data_last1['newsReplyVolum']) #评论和回复

        province_sta = data_last.groupby(data_last.index)

        data[f'sales_province_mean_{i}thMonth'] = data['model_ym'].map(province_sta.sum()['label'])
        data[f'sales_province_var_{i}thMonth'] = data['model_ym'].map(province_sta.var()['label'])
        data[f'popularity_province_mean_{i}thMonth'] = data['model_ym'].map(province_sta.sum()['popularity'])
        data[f'popularity_province_var_{i}thMonth'] = data['model_ym'].map(province_sta.var()['popularity'])

        #按车型分组统计
        data['adcode_ym_{0}'.format(i)] = data['adcode_ym'] + i
        data_last = data[~data.label.isnull()].set_index('adcode_ym_{0}'.format(i))
        model_sta = data_last.groupby(data_last.index)
        data[f'sales_model_mean_{i}thMonth'] = data['adcode_ym'].map(model_sta.sum()['label'])
        data[f'sales_model_var_{i}thMonth'] = data['adcode_ym'].map(model_sta.var()['label'])
        data[f'popularity_model_mean_{i}thMonth'] = data['adcode_ym'].map(model_sta.sum()['popularity'])
        data[f'popularity_model_var_{i}thMonth'] = data['adcode_ym'].map(model_sta.var()['popularity'])

        #按车身分组统计
        data['body_ym_{0}'.format(i)] = data['body_ym'] + i
        data_last = data[~data.label.isnull()].set_index('body_ym_{0}'.format(i))
        body_sta = data_last.groupby(data_last.index)
        data[f'sales_body_mean_{i}thMonth'] = data['body_ym'].map(body_sta.sum()['label'])
        data[f'sales_body_var_{i}thMonth'] = data['body_ym'].map(body_sta.var()['label'])
        data[f'popularity_body_mean_{i}thMonth'] = data['body_ym'].map(body_sta.sum()['popularity'])
        data[f'popularity_body_var_{i}thMonth'] = data['body_ym'].map(body_sta.var()['popularity'])

        del data['model_adcode_ym_{0}'.format(i)]
        del data['model_ym_{0}'.format(i)]
        del data['adcode_ym_{0}'.format(i)]
        del data['body_ym_{0}'.format(i)]

    # print(np.mean(data[(data['ym'] > 12) & (data['ym'] < 25)][f'sales_province_mean_12thMonth']))
    # print(np.mean(data[(data['ym'] > 12) & (data['ym'] < 25)][f'sales_model_mean_12thMonth']))
    # sys.exit(-1)
    for sea in [1, 2, 3, 4]:
        data[f'sales_{sea}thQuarter_var'] = data[[f'sales_{sea * 3 - 2}thMonth', f'sales_{sea * 3 - 1}thMonth', f'sales_{sea * 3 - 0}thMonth']].var(axis=1)
        data[f'sales_{sea}thQuarter_mean'] = data[[f'sales_{sea * 3 - 2}thMonth', f'sales_{sea * 3 - 1}thMonth', f'sales_{sea * 3 - 0}thMonth']].mean(axis=1)
        data[f'sales_model_{sea}thQuarter_var'] = data[[f'sales_model_mean_{sea * 3 - 2}thMonth', f'sales_model_mean_{sea * 3 - 1}thMonth', f'sales_model_mean_{sea * 3 - 0}thMonth']].var(axis=1)
        data[f'sales_model_{sea}thQuarter_mean'] = data[[f'sales_model_mean_{sea * 3 - 2}thMonth', f'sales_model_mean_{sea * 3 - 1}thMonth', f'sales_model_mean_{sea * 3 - 0}thMonth']].mean(axis=1)
    data[f'sales_Year_var'] = data[[f'sales_{i}thMonth' for i in range(1, 13)]].var(axis=1)
    data[f'sales_Year_mean'] = data[[f'sales_{i}thMonth' for i in range(1, 13)]].mean(axis=1)
    data[f'sales_Year_min'] = data[[f'sales_{i}thMonth' for i in range(1, 13)]].min(axis=1)
    data[f'sales_Year_max'] = data[[f'sales_{i}thMonth' for i in range(1, 13)]].max(axis=1)
    data[f'sales_model_Year_var'] = data[[f'sales_model_mean_{i}thMonth' for i in range(1, 13)]].var(axis=1)
    data[f'sales_model_Year_mean'] = data[[f'sales_model_mean_{i}thMonth' for i in range(1, 13)]].mean(axis=1)
    data[f'sales_model_Year_min'] = data[[f'sales_model_mean_{i}thMonth' for i in range(1, 13)]].min(axis=1)
    data[f'sales_model_Year_max'] = data[[f'sales_model_mean_{i}thMonth' for i in range(1, 13)]].max(axis=1)
    data[f'sales_pro_Year_var'] = data[[f'sales_province_mean_{i}thMonth' for i in range(1, 13)]].var(axis=1)
    data[f'sales_pro_Year_mean'] = data[[f'sales_province_mean_{i}thMonth' for i in range(1, 13)]].mean(axis=1)
    data[f'sales_pro_Year_min'] = data[[f'sales_province_mean_{i}thMonth' for i in range(1, 13)]].min(axis=1)
    data[f'sales_pro_Year_max'] = data[[f'sales_province_mean_{i}thMonth' for i in range(1, 13)]].max(axis=1)
    data[f'sales_body_Year_var'] = data[[f'sales_body_mean_{i}thMonth' for i in range(1, 13)]].var(axis=1)
    data[f'sales_body_Year_mean'] = data[[f'sales_body_mean_{i}thMonth' for i in range(1, 13)]].mean(axis=1)
    data[f'sales_body_Year_min'] = data[[f'sales_body_mean_{i}thMonth' for i in range(1, 13)]].min(axis=1)
    data[f'sales_body_Year_max'] = data[[f'sales_body_mean_{i}thMonth' for i in range(1, 13)]].max(axis=1)
    #窗口和历史特征，窗口大小分别是3 4 5 6 7 12

    # for win_size in [2,3,4,5,6,7,8,9,10,11,12]:
    #     for fea in ['sales_', 'popularity_', 'comment_', 'reply_',
    #                 'sales_province_mean_', 'sales_province_var_', 'popularity_province_mean_', 'popularity_province_var_',
    #                 'sales_model_mean_', 'sales_model_var_', 'popularity_model_mean_', 'popularity_model_var_',
    #                 'sales_body_mean_', 'sales_body_var_', 'popularity_body_mean_', 'popularity_body_var_']:
    #         window_fea = [fea+f'{i}thMonth' for i in range(1,win_size+1)]
    #         data[fea+f'window1_var_{win_size}'] = data[window_fea].var(axis=1)
    #         data[fea+f'window1_mean_{win_size}'] = data[window_fea].mean(axis=1)

            #window_fea = [fea + f'{i}thMonth' for i in range(1, win_size)]
            #data[fea + f'window1_mean_{win_size}'] = data[window_fea].mean(axis=1)
    # for win_size in [2,3,4,5,6,7,8,9,10,11,12]:
    #     for fea in ['sales_', 'popularity_', 'comment_', 'reply_',
    #                 'sales_province_mean_', 'sales_province_var_', 'popularity_province_mean_', 'popularity_province_var_',
    #                 'sales_model_mean_', 'sales_model_var_', 'popularity_model_mean_', 'popularity_model_var_',
    #                 'sales_body_mean_', 'sales_body_var_', 'popularity_body_mean_', 'popularity_body_var_']:
    #         window_fea = [fea+f'{i}thMonth' for i in range(2,win_size+1)]
    #         data[fea+f'window2_var_{win_size}'] = data[window_fea].var(axis=1)
    #         data[fea+f'window2_mean_{win_size}'] = data[window_fea].mean(axis=1)
    #
    # for win_size in [2,3,4,5,6,7,8,9,10,11,12]:
    #     for fea in ['sales_', 'popularity_', 'comment_', 'reply_',
    #                 'sales_province_mean_', 'sales_province_var_', 'popularity_province_mean_', 'popularity_province_var_',
    #                 'sales_model_mean_', 'sales_model_var_', 'popularity_model_mean_', 'popularity_model_var_',
    #                 'sales_body_mean_', 'sales_body_var_', 'popularity_body_mean_', 'popularity_body_var_']:
    #         window_fea = [fea+f'{i}thMonth' for i in range(3,win_size+1)]
    #         data[fea+f'window3_var_{win_size}'] = data[window_fea].var(axis=1)
    #         data[fea+f'window3_mean_{win_size}'] = data[window_fea].mean(axis=1)
    #
    # for win_size in [2,3,4,5,6,7,8,9,10,11,12]:
    #     for fea in ['sales_', 'popularity_', 'comment_', 'reply_',
    #                 'sales_province_mean_', 'sales_province_var_', 'popularity_province_mean_', 'popularity_province_var_',
    #                 'sales_model_mean_', 'sales_model_var_', 'popularity_model_mean_', 'popularity_model_var_',
    #                 'sales_body_mean_', 'sales_body_var_', 'popularity_body_mean_', 'popularity_body_var_']:
    #         window_fea = [fea+f'{i}thMonth' for i in range(4,win_size+1)]
    #         data[fea+f'window4_var_{win_size}'] = data[window_fea].var(axis=1)
    #         data[fea+f'window4_mean_{win_size}'] = data[window_fea].mean(axis=1)
    #趋势特征:一阶
    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        j = i+1
        for fea in ['sales_', 'popularity_', 'comment_', 'reply_',
                    'sales_province_mean_', 'sales_province_var_', 'popularity_province_mean_', 'popularity_province_var_',
                    'sales_model_mean_', 'sales_model_var_', 'popularity_model_mean_', 'popularity_model_var_',
                    'sales_body_mean_', 'sales_body_var_', 'popularity_body_mean_', 'popularity_body_var_']:
            data[fea+f'_diff_{i}_{j}'] = data[fea+f'{i}thMonth'] - data[fea+f'{j}thMonth']
            data[fea+f'_time_{i}_{j}'] = data[fea+f'{i}thMonth'] / data[fea+f'{j}thMonth']

    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        j = i+1
        for fea in ['sales_', 'popularity_', 'comment_', 'reply_',
                    'sales_province_mean_', 'sales_province_var_', 'popularity_province_mean_', 'popularity_province_var_',
                    'sales_model_mean_', 'sales_model_var_', 'popularity_model_mean_', 'popularity_model_var_',
                    'sales_body_mean_', 'sales_body_var_', 'popularity_body_mean_', 'popularity_body_var_']:
            data[fea+f'_diff_{i}_{j}'] = data[fea+f'{i}thMonth'] - data[fea+f'{j}thMonth']
            data[fea+f'_time_{i}_{j}'] = data[fea+f'{i}thMonth'] / data[fea+f'{j}thMonth']

    # 趋势特征:二阶
    # for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
    #     j = i+1
    #     k = i+2
    #     for fea in ['sales_', 'popularity_', 'comment_', 'reply_',
    #                 'sales_province_mean_', 'sales_province_var_', 'popularity_province_mean_', 'popularity_province_var_',
    #                 'sales_model_mean_', 'sales_model_var_', 'popularity_model_mean_', 'popularity_model_var_',
    #                 'sales_body_mean_', 'sales_body_var_', 'popularity_body_mean_', 'popularity_body_var_']:
    #         data[fea+f'_diff_{i}_{j}2'] = data[fea+f'_diff_{i}_{j}'] - data[fea+f'_diff_{j}_{k}']
    #         data[fea+f'_time_{i}_{j}2'] = data[fea+f'_diff_{i}_{j}'] / data[fea+f'_diff_{j}_{k}']




    #print(data.info())
    with open(P_DATA + 'train_columns.txt', 'w') as train_f:
        for n in data.columns:
            train_f.write(str(n)+'\n')
    with open(P_DATA + 'train.pk', 'wb') as train_f:
        pickle.dump(data, train_f)
    print(f'feature process use time {time.time()-t}')
if __name__ == "__main__":
    t1 = time.time()
    feature_main()
    print('特征处理时间',time.time()-t1)
