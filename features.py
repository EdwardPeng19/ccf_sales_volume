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
    model_bodyType = train_sales_model.groupby(['model','bodyType']).size().reset_index(name='count')
    del model_bodyType['count']
    evaluation_public = evaluation_public.merge(model_bodyType, how='left', on=['model'])

    #data_describe(train_search, ['province', 'adcode', 'model', 'regYear', 'regMonth'])
    #data_describe(train_sales_model, ['province', 'adcode', 'model', 'bodyType', 'regYear', 'regMonth'])
    #data_describe(train_user_reply, ['model', 'regYear', 'regMonth'])

    data = pd.concat([train_sales_model, evaluation_public], ignore_index=True)
    data = data.merge(train_search, how='left', on=['province', 'adcode', 'model', 'regYear', 'regMonth'])
    data = data.merge(train_user_reply, how='left', on=['model', 'regYear', 'regMonth'])
    data['id'] = data['id'].fillna(0).astype(int)
    data['label'] = data['salesVolume']
    del data['salesVolume'], data['forecastVolum']
    #print(data.info())
    return data
    # with open(P_DATA + 'train_pre.pk', 'wb') as train_f:
    #     pickle.dump(data, train_f)
    # with open(P_DATA + 'test.pk', 'wb') as test_f:
    #     pickle.dump(test, test_f)

#统计特征
def feature_count(sales_model, search, reply):
    sales_model['season'] = sales_model['regMonth']%4
    season_sales = sales_model.groupby('season')['salesVolume'].agg([np.mean, np.var]).add_prefix('season_sales_').reset_index()

    #province_sales_model = sales_model[['province','salesVolume']].groupby('province').mean().add_prefix('mean_province_').reset_index()

    province_sales_model = sales_model.groupby('province')['salesVolume'].agg([np.mean, np.var]).add_prefix('province_sales_').reset_index()


    model_sales_model = sales_model[['model', 'salesVolume']].groupby('model').mean().add_prefix('mean_model_').reset_index()
    bodyType_sales_model = sales_model[['bodyType', 'salesVolume']].groupby('bodyType').mean().add_prefix('mean_bodyType_').reset_index()
    province_sales_model2 = sales_model[['province', 'regMonth', 'salesVolume']].groupby(['province', 'regMonth']).mean().add_prefix('mean_province_permonth_').reset_index()
    model_sales_model2 = sales_model[['model', 'regMonth', 'salesVolume']].groupby(['model','regMonth']).mean().add_prefix('mean_model_permonth_').reset_index()
    bodyType_sales_model2 = sales_model[['bodyType', 'regMonth', 'salesVolume']].groupby(['bodyType','regMonth']).mean().add_prefix('mean_bodyType_permonth_').reset_index()



    province_search = search[['province', 'popularity']].groupby('province').mean().add_prefix('mean_province_').reset_index()
    model_search = search[['model', 'popularity']].groupby('model').mean().add_prefix('mean_model_').reset_index()
    province_search2 = search[['province', 'regMonth', 'popularity']].groupby(['province','regMonth']).mean().add_prefix('mean_province_permonth_').reset_index()
    model_search2 = search[['model', 'regMonth', 'popularity']].groupby(['model','regMonth']).mean().add_prefix('mean_model_permonth_').reset_index()

    model_reply1 = reply[['model', 'newsReplyVolum']].groupby('model').mean().add_prefix('mean_model_').reset_index()
    model_reply2 = reply[['model', 'carCommentVolum']].groupby('model').mean().add_prefix('mean_model_').reset_index()
    model_reply3 = reply[['model', 'regMonth','newsReplyVolum']].groupby(['model','regMonth']).mean().add_prefix('mean_model_permonth_').reset_index()
    model_reply4 = reply[['model', 'regMonth','carCommentVolum']].groupby(['model','regMonth']).mean().add_prefix('mean_model_permonth_').reset_index()

    return province_sales_model,model_sales_model,bodyType_sales_model, province_search,model_search,model_reply1,model_reply2,\
           province_sales_model2,model_sales_model2,bodyType_sales_model2, province_search2,model_search2,model_reply3,model_reply4,\
           season_sales


def window_rollth(x):
    return list(x)[0]


def feature_main(sales_path=None):
    data = train_input()
    #窗口平移 #'popularity', 'carCommentVolum', 'newsReplyVolum'
    data['season'] = data['regMonth'] % 4
    # 统计特征
    train_sales = pd.read_csv(P_DATA + 'train_sales_data.csv', encoding='utf-8')
    train_search = pd.read_csv(P_DATA + 'train_search_data.csv', encoding='utf-8')
    train_user_reply = pd.read_csv(P_DATA + 'train_user_reply_data.csv', encoding='utf-8')
    province_sales,model_sales,bodyType_sales, \
    province_search,model_search,\
    model_reply1,model_reply2, \
    province_sales2, model_sales2, bodyType_sales2, \
    province_search2, model_search2, \
    model_reply3, model_reply4, season_sales \
    = feature_count(train_sales, train_search, train_user_reply)

    #   每个季度的销量


    data = data.merge(season_sales, how='left', on=['season'])
    data = data.merge(province_sales, how='left', on=['province'])
    data = data.merge(model_sales, how='left', on=['model'])
    data = data.merge(bodyType_sales, how='left', on=['bodyType'])
    data = data.merge(province_search, how='left', on=['province'])
    data = data.merge(model_search, how='left', on=['model'])
    data = data.merge(model_reply1, how='left', on=['model'])
    data = data.merge(model_reply2, how='left', on=['model'])

    data = data.merge(province_sales2, how='left', on=['province', 'regMonth'])
    data = data.merge(model_sales2, how='left', on=['model', 'regMonth'])
    data = data.merge(bodyType_sales2, how='left', on=['bodyType', 'regMonth'])
    data = data.merge(province_search2, how='left', on=['province', 'regMonth'])
    data = data.merge(model_search2, how='left', on=['model', 'regMonth'])
    data = data.merge(model_reply3, how='left', on=['model', 'regMonth'])
    data = data.merge(model_reply4, how='left', on=['model', 'regMonth'])
    print(data.columns)

    data['label'] = data['label'].fillna(0)
    data['popularity'] = data['popularity'].fillna(0)
    data['carCommentVolum'] = data['carCommentVolum'].fillna(0)
    data['newsReplyVolum'] = data['newsReplyVolum'].fillna(0)
    if sales_path:
        sales_data = pd.read_csv(sales_path).rename(columns={'forecastVolum':'salesNum'})
        data = data.merge(sales_data, how='left', on='id')
        data['label'] = data.apply(lambda x: x['salesNum'] if x['salesNum']>0 else x['label'] , axis=1)


    data['mp_fea'] = data['adcode'].astype(str) + data['model']
    data['id_index'] = range(len(data))
    redf = pd.DataFrame()
    #models = data['model'].value_counts().index
    #models_df = pd.DataFrame(list(models),columns=['model']).to_csv('models.csv', index=False)
    models = pd.read_csv('models.csv')['model']
    #provinces = data['province'].value_counts().index
    #provinces_df = pd.DataFrame(list(provinces), columns=['province']).to_csv('provinces.csv', index=False)
    provinces = pd.read_csv('provinces.csv')['province']
    #bodys = data['bodyType'].value_counts().index
    #bodys_df = pd.DataFrame(list(bodys), columns=['body']).to_csv('bodys.csv', index=False)
    bodys = pd.read_csv('bodys.csv')['body']

    print('m p 特征')
    data.sort_values(by=['province', 'model', 'regYear', 'regMonth'], ascending=[True, True, True, True], inplace=True)
    for th in range(2, 14):
        data[f'sales_{th - 1}thMonth'] = data.rolling(th)["label"].apply(lambda x: list(x)[0], raw=False)  # 往前第th个月的销量
        data[f'popularity_{th - 1}thMonth'] = data.rolling(th)["popularity"].apply(lambda x: list(x)[0], raw=False)  # 往前第th个月的销量

    #对车型相关新闻文章的评论数量 carCommentVolum 和对车型的评价数量的回溯 newsReplyVolum
    print('m 特征')
    redf = pd.DataFrame()
    for m in models:
        m_df = data[(data['model'] == m)]
        m_df.sort_values(by=['regYear', 'regMonth'], ascending=[True, True], inplace=True)
        n_df = m_df[['regYear', 'regMonth', 'label']].groupby(['regYear', 'regMonth']).sum().add_prefix('allp_').reset_index()
        p_df = m_df[['regYear', 'regMonth', 'popularity']].groupby(['regYear', 'regMonth']).sum().add_prefix('allp_').reset_index()

        c_df = m_df[['regYear', 'regMonth', 'carCommentVolum']].groupby(['regYear', 'regMonth']).mean().add_prefix('allc_').reset_index()
        r_df = m_df[['regYear', 'regMonth', 'newsReplyVolum']].groupby(['regYear', 'regMonth']).mean().add_prefix('allr_').reset_index()

        for th in range(2,14):
            n_df[f'sales_model_{th-1}thMonth'] = n_df.rolling(th)["allp_label"].apply(window_rollth, raw=False) # 往前第th个月的销量
            p_df[f'popularity_model_{th - 1}thMonth'] = p_df.rolling(th)["allp_popularity"].apply(window_rollth, raw=False)  # 往前第th个月的销量
            c_df[f'comment_{th - 1}thMonth'] = c_df.rolling(th)["allc_carCommentVolum"].apply(window_rollth, raw=False)  # 往前第th个月的评论特征
            r_df[f'reply_{th - 1}thMonth'] = r_df.rolling(th)["allr_newsReplyVolum"].apply(window_rollth, raw=False)  # 往前第th个月的评论特征
        m_df = m_df.merge(n_df, how='left', on=['regYear', 'regMonth'])
        m_df = m_df.merge(p_df, how='left', on=['regYear', 'regMonth'])
        m_df = m_df.merge(c_df, how='left', on=['regYear', 'regMonth'])
        m_df = m_df.merge(r_df, how='left', on=['regYear', 'regMonth'])

        redf = redf.append(m_df)
    data = redf

    #以省分组
    print('p 特征')
    redf = pd.DataFrame()
    for p in provinces:
        m_df = data[(data['province'] == p)]
        m_df.sort_values(by=['regYear', 'regMonth'], ascending=[True, True], inplace=True)
        m_df.sort_values(by=['regYear', 'regMonth'], ascending=[True, True], inplace=True)
        n_df = m_df[['regYear', 'regMonth', 'label']].groupby(['regYear', 'regMonth']).sum().add_prefix('allm_').reset_index()
        p_df = m_df[['regYear', 'regMonth', 'popularity']].groupby(['regYear', 'regMonth']).sum().add_prefix('allm_').reset_index()
        n_df['sales_pro_cumsum'] = n_df['allm_label'].cumsum() - n_df['allm_label']
        p_df['popularity_pro_cumsum'] = p_df['allm_popularity'].cumsum() - p_df['allm_popularity']

        for th in range(2, 14):
            n_df[f'sales_pro_{th - 1}thMonth'] = n_df.rolling(th)["allm_label"].apply(window_rollth, raw=False)  # 往前第th个月的销量
            p_df[f'popularity_pro_{th - 1}thMonth'] = p_df.rolling(th)["allm_popularity"].apply(window_rollth,raw=False)  # 往前第th个月的搜索量
        m_df = m_df.merge(n_df, how='left', on=['regYear', 'regMonth'])
        m_df = m_df.merge(p_df, how='left', on=['regYear', 'regMonth'])
        redf = redf.append(m_df)
    data = redf

    #以车型分组
    print('b 特征')
    redf = pd.DataFrame()
    for b in bodys:
        m_df = data[(data['bodyType'] == b)]
        m_df.sort_values(by=['regYear', 'regMonth'], ascending=[True, True], inplace=True)
        m_df.sort_values(by=['regYear', 'regMonth'], ascending=[True, True], inplace=True)
        n_df = m_df[['regYear', 'regMonth', 'label']].groupby(['regYear', 'regMonth']).sum().add_prefix('allb_').reset_index()
        p_df = m_df[['regYear', 'regMonth', 'popularity']].groupby(['regYear', 'regMonth']).sum().add_prefix('allb_').reset_index()
        n_df['sales_body_cumsum'] = n_df['allb_label'].cumsum() - n_df['allb_label']
        p_df['popularity_body_cumsum'] = p_df['allb_popularity'].cumsum() - p_df['allb_popularity']

        for th in range(2, 14):
            n_df[f'sales_body_{th - 1}thMonth'] = n_df.rolling(th)["allb_label"].apply(window_rollth, raw=False)  # 往前第th个月的销量
            p_df[f'popularity_body_{th - 1}thMonth'] = p_df.rolling(th)["allb_popularity"].apply(window_rollth,raw=False)  # 往前第th个月的搜索量
        m_df = m_df.merge(n_df, how='left', on=['regYear', 'regMonth'])
        m_df = m_df.merge(p_df, how='left', on=['regYear', 'regMonth'])
        redf = redf.append(m_df)
    data = redf

    for sea in [1, 2, 3, 4]:
        data[f'sales_{sea}thQuarter_var'] = data[[f'sales_{sea * 3 - 2}thMonth', f'sales_{sea * 3 - 1}thMonth', f'sales_{sea * 3 - 0}thMonth']].var(axis=1)
        data[f'sales_{sea}thQuarter_std'] = data[[f'sales_{sea * 3 - 2}thMonth', f'sales_{sea * 3 - 1}thMonth', f'sales_{sea * 3 - 0}thMonth']].std(axis=1)
        data[f'sales_{sea}thQuarter_mean'] = data[[f'sales_{sea * 3 - 2}thMonth', f'sales_{sea * 3 - 1}thMonth', f'sales_{sea * 3 - 0}thMonth']].mean(axis=1)
        data[f'sales_{sea}thQuarter_min'] = data[[f'sales_{sea * 3 - 2}thMonth', f'sales_{sea * 3 - 1}thMonth', f'sales_{sea * 3 - 0}thMonth']].min(axis=1)
        data[f'sales_{sea}thQuarter_max'] = data[[f'sales_{sea * 3 - 2}thMonth', f'sales_{sea * 3 - 1}thMonth', f'sales_{sea * 3 - 0}thMonth']].max(axis=1)
        data[f'sales_model_{sea}thQuarter_var'] = data[[f'sales_model_{sea * 3 - 2}thMonth', f'sales_model_{sea * 3 - 1}thMonth', f'sales_model_{sea * 3 - 0}thMonth']].var(axis=1)
        data[f'sales_model_{sea}thQuarter_mean'] = data[[f'sales_model_{sea * 3 - 2}thMonth', f'sales_model_{sea * 3 - 1}thMonth', f'sales_model_{sea * 3 - 0}thMonth']].mean(axis=1)
        data[f'sales_model_{sea}thQuarter_min'] = data[[f'sales_model_{sea * 3 - 2}thMonth', f'sales_model_{sea * 3 - 1}thMonth', f'sales_model_{sea * 3 - 0}thMonth']].min(axis=1)
        data[f'sales_model_{sea}thQuarter_max'] = data[[f'sales_model_{sea * 3 - 2}thMonth', f'sales_model_{sea * 3 - 1}thMonth', f'sales_model_{sea * 3 - 0}thMonth']].max(axis=1)
        data[f'sales_pro_{sea}thQuarter_var'] = data[[f'sales_pro_{sea * 3 - 2}thMonth', f'sales_pro_{sea * 3 - 1}thMonth', f'sales_pro_{sea * 3 - 0}thMonth']].var(axis=1)
        data[f'sales_pro_{sea}thQuarter_mean'] = data[[f'sales_pro_{sea * 3 - 2}thMonth', f'sales_pro_{sea * 3 - 1}thMonth', f'sales_pro_{sea * 3 - 0}thMonth']].mean(axis=1)
        data[f'sales_pro_{sea}thQuarter_min'] = data[[f'sales_pro_{sea * 3 - 2}thMonth', f'sales_pro_{sea * 3 - 1}thMonth', f'sales_pro_{sea * 3 - 0}thMonth']].min(axis=1)
        data[f'sales_pro_{sea}thQuarter_max'] = data[[f'sales_pro_{sea * 3 - 2}thMonth', f'sales_pro_{sea * 3 - 1}thMonth', f'sales_pro_{sea * 3 - 0}thMonth']].max(axis=1)
        data[f'sales_body_{sea}thQuarter_var'] = data[[f'sales_body_{sea * 3 - 2}thMonth', f'sales_body_{sea * 3 - 1}thMonth', f'sales_body_{sea * 3 - 0}thMonth']].var(axis=1)
        data[f'sales_body_{sea}thQuarter_mean'] = data[[f'sales_body_{sea * 3 - 2}thMonth', f'sales_body_{sea * 3 - 1}thMonth', f'sales_body_{sea * 3 - 0}thMonth']].mean(axis=1)
        data[f'sales_body_{sea}thQuarter_min'] = data[[f'sales_body_{sea * 3 - 2}thMonth', f'sales_body_{sea * 3 - 1}thMonth', f'sales_body_{sea * 3 - 0}thMonth']].min(axis=1)
        data[f'sales_body_{sea}thQuarter_max'] = data[[f'sales_body_{sea * 3 - 2}thMonth', f'sales_body_{sea * 3 - 1}thMonth', f'sales_body_{sea * 3 - 0}thMonth']].max(axis=1)

        data[f'popularity_{sea}thQuarter_var'] = data[[f'popularity_{sea * 3 - 2}thMonth', f'popularity_{sea * 3 - 1}thMonth', f'popularity_{sea * 3 - 0}thMonth']].var(axis=1)
        data[f'popularity_{sea}thQuarter_mean'] = data[[f'popularity_{sea * 3 - 2}thMonth', f'popularity_{sea * 3 - 1}thMonth', f'popularity_{sea * 3 - 0}thMonth']].mean(axis=1)
        data[f'popularity_{sea}thQuarter_min'] = data[[f'popularity_{sea * 3 - 2}thMonth', f'popularity_{sea * 3 - 1}thMonth', f'popularity_{sea * 3 - 0}thMonth']].min(axis=1)
        data[f'popularity_{sea}thQuarter_max'] = data[[f'popularity_{sea * 3 - 2}thMonth', f'popularity_{sea * 3 - 1}thMonth', f'popularity_{sea * 3 - 0}thMonth']].max(axis=1)
        data[f'popularity_model_{sea}thQuarter_var'] = data[[f'popularity_model_{sea * 3 - 2}thMonth', f'popularity_model_{sea * 3 - 1}thMonth', f'popularity_model_{sea * 3 - 0}thMonth']].var(axis=1)
        data[f'popularity_model_{sea}thQuarter_mean'] = data[[f'popularity_model_{sea * 3 - 2}thMonth', f'popularity_model_{sea * 3 - 1}thMonth', f'popularity_model_{sea * 3 - 0}thMonth']].mean(axis=1)
        data[f'popularity_model_{sea}thQuarter_min'] = data[[f'popularity_model_{sea * 3 - 2}thMonth', f'popularity_model_{sea * 3 - 1}thMonth', f'popularity_model_{sea * 3 - 0}thMonth']].min(axis=1)
        data[f'popularity_model_{sea}thQuarter_max'] = data[[f'popularity_model_{sea * 3 - 2}thMonth', f'popularity_model_{sea * 3 - 1}thMonth', f'popularity_model_{sea * 3 - 0}thMonth']].max(axis=1)
        data[f'popularity_pro_{sea}thQuarter_var'] = data[[f'popularity_pro_{sea * 3 - 2}thMonth', f'popularity_pro_{sea * 3 - 1}thMonth', f'popularity_pro_{sea * 3 - 0}thMonth']].var(axis=1)
        data[f'popularity_pro_{sea}thQuarter_mean'] = data[[f'popularity_pro_{sea * 3 - 2}thMonth', f'popularity_pro_{sea * 3 - 1}thMonth', f'popularity_pro_{sea * 3 - 0}thMonth']].mean(axis=1)
        data[f'popularity_pro_{sea}thQuarter_min'] = data[[f'popularity_pro_{sea * 3 - 2}thMonth', f'popularity_pro_{sea * 3 - 1}thMonth', f'popularity_pro_{sea * 3 - 0}thMonth']].min(axis=1)
        data[f'popularity_pro_{sea}thQuarter_max'] = data[[f'popularity_pro_{sea * 3 - 2}thMonth', f'popularity_pro_{sea * 3 - 1}thMonth', f'popularity_pro_{sea * 3 - 0}thMonth']].max(axis=1)
        data[f'popularity_body_{sea}thQuarter_var'] = data[[f'popularity_body_{sea * 3 - 2}thMonth', f'popularity_body_{sea * 3 - 1}thMonth', f'popularity_body_{sea * 3 - 0}thMonth']].var(axis=1)
        data[f'popularity_body_{sea}thQuarter_mean'] = data[[f'popularity_body_{sea * 3 - 2}thMonth', f'popularity_body_{sea * 3 - 1}thMonth', f'popularity_body_{sea * 3 - 0}thMonth']].mean(axis=1)
        data[f'popularity_body_{sea}thQuarter_min'] = data[[f'popularity_body_{sea * 3 - 2}thMonth', f'popularity_body_{sea * 3 - 1}thMonth', f'popularity_body_{sea * 3 - 0}thMonth']].min(axis=1)
        data[f'popularity_body_{sea}thQuarter_max'] = data[[f'popularity_body_{sea * 3 - 2}thMonth', f'popularity_body_{sea * 3 - 1}thMonth', f'popularity_body_{sea * 3 - 0}thMonth']].max(axis=1)

        data[f'comment_{sea}thQuarter_var'] = data[[f'comment_{sea * 3 - 2}thMonth', f'comment_{sea * 3 - 1}thMonth', f'comment_{sea * 3 - 0}thMonth']].var(axis=1)
        data[f'comment_{sea}thQuarter_mean'] = data[[f'comment_{sea * 3 - 2}thMonth', f'comment_{sea * 3 - 1}thMonth', f'comment_{sea * 3 - 0}thMonth']].mean(axis=1)
        data[f'comment_{sea}thQuarter_min'] = data[[f'comment_{sea * 3 - 2}thMonth', f'comment_{sea * 3 - 1}thMonth', f'comment_{sea * 3 - 0}thMonth']].min(axis=1)
        data[f'comment_{sea}thQuarter_max'] = data[[f'comment_{sea * 3 - 2}thMonth', f'comment_{sea * 3 - 1}thMonth', f'comment_{sea * 3 - 0}thMonth']].max(axis=1)
        data[f'reply_{sea}thQuarter_var'] = data[[f'reply_{sea * 3 - 2}thMonth', f'reply_{sea * 3 - 1}thMonth', f'reply_{sea * 3 - 0}thMonth']].var(axis=1)
        data[f'reply_{sea}thQuarter_mean'] = data[[f'reply_{sea * 3 - 2}thMonth', f'reply_{sea * 3 - 1}thMonth', f'reply_{sea * 3 - 0}thMonth']].mean(axis=1)
        data[f'reply_{sea}thQuarter_min'] = data[[f'reply_{sea * 3 - 2}thMonth', f'reply_{sea * 3 - 1}thMonth', f'reply_{sea * 3 - 0}thMonth']].min(axis=1)
        data[f'reply_{sea}thQuarter_max'] = data[[f'reply_{sea * 3 - 2}thMonth', f'reply_{sea * 3 - 1}thMonth', f'reply_{sea * 3 - 0}thMonth']].max(axis=1)
    for hy in [1, 2]:
        data[f'sales_{hy}thHalfyear_var'] = data[[f'sales_{hy * 6 - i}thMonth' for i in range(6)]].var(axis=1)
        data[f'sales_{hy}thHalfyear_mean'] = data[[f'sales_{hy * 6 - i}thMonth' for i in range(6)]].mean(axis=1)
        data[f'sales_{hy}thHalfyear_min'] = data[[f'sales_{hy * 6 - i}thMonth' for i in range(6)]].min(axis=1)
        data[f'sales_{hy}thHalfyear_max'] = data[[f'sales_{hy * 6 - i}thMonth' for i in range(6)]].max(axis=1)
        data[f'sales_model_{hy}thHalfyear_var'] = data[[f'sales_model_{hy * 6 - i}thMonth' for i in range(6)]].var(axis=1)
        data[f'sales_model_{hy}thHalfyear_mean'] = data[[f'sales_model_{hy * 6 - i}thMonth' for i in range(6)]].mean(axis=1)
        data[f'sales_model_{hy}thHalfyear_min'] = data[[f'sales_model_{hy * 6 - i}thMonth' for i in range(6)]].min(axis=1)
        data[f'sales_model_{hy}thHalfyear_max'] = data[[f'sales_model_{hy * 6 - i}thMonth' for i in range(6)]].max(axis=1)
        data[f'sales_pro_{hy}thHalfyear_var'] = data[[f'sales_pro_{hy * 6 - i}thMonth' for i in range(6)]].var(axis=1)
        data[f'sales_pro_{hy}thHalfyear_mean'] = data[[f'sales_pro_{hy * 6 - i}thMonth' for i in range(6)]].mean(axis=1)
        data[f'sales_pro_{hy}thHalfyear_min'] = data[[f'sales_pro_{hy * 6 - i}thMonth' for i in range(6)]].min(axis=1)
        data[f'sales_pro_{hy}thHalfyear_max'] = data[[f'sales_pro_{hy * 6 - i}thMonth' for i in range(6)]].max(axis=1)
        data[f'sales_body_{hy}thHalfyear_var'] = data[[f'sales_body_{hy * 6 - i}thMonth' for i in range(6)]].var(axis=1)
        data[f'sales_body_{hy}thHalfyear_mean'] = data[[f'sales_body_{hy * 6 - i}thMonth' for i in range(6)]].mean(axis=1)
        data[f'sales_body_{hy}thHalfyear_min'] = data[[f'sales_body_{hy * 6 - i}thMonth' for i in range(6)]].min(axis=1)
        data[f'sales_body_{hy}thHalfyear_max'] = data[[f'sales_body_{hy * 6 - i}thMonth' for i in range(6)]].max(axis=1)

        data[f'popularity_{hy}thHalfyear_var'] = data[[f'popularity_{hy * 6 - i}thMonth' for i in range(6)]].var(axis=1)
        data[f'popularity_{hy}thHalfyear_mean'] = data[[f'popularity_{hy * 6 - i}thMonth' for i in range(6)]].mean(axis=1)
        data[f'popularity_{hy}thHalfyear_min'] = data[[f'popularity_{hy * 6 - i}thMonth' for i in range(6)]].min(axis=1)
        data[f'popularity_{hy}thHalfyear_max'] = data[[f'popularity_{hy * 6 - i}thMonth' for i in range(6)]].max(axis=1)
        data[f'popularity_model_{hy}thHalfyear_var'] = data[[f'popularity_model_{hy * 6 - i}thMonth' for i in range(6)]].var(axis=1)
        data[f'popularity_model_{hy}thHalfyear_mean'] = data[[f'popularity_model_{hy * 6 - i}thMonth' for i in range(6)]].mean(axis=1)
        data[f'popularity_model_{hy}thHalfyear_min'] = data[[f'popularity_model_{hy * 6 - i}thMonth' for i in range(6)]].min(axis=1)
        data[f'popularity_model_{hy}thHalfyear_max'] = data[[f'popularity_model_{hy * 6 - i}thMonth' for i in range(6)]].max(axis=1)
        data[f'popularity_pro_{hy}thHalfyear_var'] = data[[f'popularity_pro_{hy * 6 - i}thMonth' for i in range(6)]].var(axis=1)
        data[f'popularity_pro_{hy}thHalfyear_mean'] = data[[f'popularity_pro_{hy * 6 - i}thMonth' for i in range(6)]].mean(axis=1)
        data[f'popularity_pro_{hy}thHalfyear_min'] = data[[f'popularity_pro_{hy * 6 - i}thMonth' for i in range(6)]].min(axis=1)
        data[f'popularity_pro_{hy}thHalfyear_max'] = data[[f'popularity_pro_{hy * 6 - i}thMonth' for i in range(6)]].max(axis=1)
        data[f'popularity_body_{hy}thHalfyear_var'] = data[[f'popularity_body_{hy * 6 - i}thMonth' for i in range(6)]].var(axis=1)
        data[f'popularity_body_{hy}thHalfyear_mean'] = data[[f'popularity_body_{hy * 6 - i}thMonth' for i in range(6)]].mean(axis=1)
        data[f'popularity_body_{hy}thHalfyear_min'] = data[[f'popularity_body_{hy * 6 - i}thMonth' for i in range(6)]].min(axis=1)
        data[f'popularity_body_{hy}thHalfyear_max'] = data[[f'popularity_body_{hy * 6 - i}thMonth' for i in range(6)]].max(axis=1)

        data[f'comment_{hy}thHalfyear_var'] = data[[f'comment_{hy * 6 - i}thMonth' for i in range(6)]].var(axis=1)
        data[f'comment_{hy}thHalfyear_mean'] = data[[f'comment_{hy * 6 - i}thMonth' for i in range(6)]].mean(axis=1)
        data[f'comment_{hy}thHalfyear_min'] = data[[f'comment_{hy * 6 - i}thMonth' for i in range(6)]].min(axis=1)
        data[f'comment_{hy}thHalfyear_max'] = data[[f'comment_{hy * 6 - i}thMonth' for i in range(6)]].max(axis=1)
        data[f'reply_{hy}thHalfyear_var'] = data[[f'reply_{hy * 6 - i}thMonth' for i in range(6)]].var(axis=1)
        data[f'reply_{hy}thHalfyear_mean'] = data[[f'reply_{hy * 6 - i}thMonth' for i in range(6)]].mean(axis=1)
        data[f'reply_{hy}thHalfyear_min'] = data[[f'reply_{hy * 6 - i}thMonth' for i in range(6)]].min(axis=1)
        data[f'reply_{hy}thHalfyear_max'] = data[[f'reply_{hy * 6 - i}thMonth' for i in range(6)]].max(axis=1)
    data[f'sales_Year_var'] = data[[f'sales_{i}thMonth' for i in range(1, 13)]].var(axis=1)
    data[f'sales_Year_mean'] = data[[f'sales_{i}thMonth' for i in range(1, 13)]].mean(axis=1)
    data[f'sales_Year_min'] = data[[f'sales_{i}thMonth' for i in range(1, 13)]].min(axis=1)
    data[f'sales_Year_max'] = data[[f'sales_{i}thMonth' for i in range(1, 13)]].max(axis=1)
    data[f'sales_model_Year_var'] = data[[f'sales_model_{i}thMonth' for i in range(1, 13)]].var(axis=1)
    data[f'sales_model_Year_mean'] = data[[f'sales_model_{i}thMonth' for i in range(1, 13)]].mean(axis=1)
    data[f'sales_model_Year_min'] = data[[f'sales_model_{i}thMonth' for i in range(1, 13)]].min(axis=1)
    data[f'sales_model_Year_max'] = data[[f'sales_model_{i}thMonth' for i in range(1, 13)]].max(axis=1)
    data[f'sales_pro_Year_var'] = data[[f'sales_pro_{i}thMonth' for i in range(1, 13)]].var(axis=1)
    data[f'sales_pro_Year_mean'] = data[[f'sales_pro_{i}thMonth' for i in range(1, 13)]].mean(axis=1)
    data[f'sales_pro_Year_min'] = data[[f'sales_pro_{i}thMonth' for i in range(1, 13)]].min(axis=1)
    data[f'sales_pro_Year_max'] = data[[f'sales_pro_{i}thMonth' for i in range(1, 13)]].max(axis=1)
    data[f'sales_body_Year_var'] = data[[f'sales_body_{i}thMonth' for i in range(1, 13)]].var(axis=1)
    data[f'sales_body_Year_mean'] = data[[f'sales_body_{i}thMonth' for i in range(1, 13)]].mean(axis=1)
    data[f'sales_body_Year_min'] = data[[f'sales_body_{i}thMonth' for i in range(1, 13)]].min(axis=1)
    data[f'sales_body_Year_max'] = data[[f'sales_body_{i}thMonth' for i in range(1, 13)]].max(axis=1)

    data[f'popularity_Year_var'] = data[[f'popularity_{i}thMonth' for i in range(1, 13)]].var(axis=1)
    data[f'popularity_Year_mean'] = data[[f'popularity_{i}thMonth' for i in range(1, 13)]].mean(axis=1)
    data[f'popularity_Year_min'] = data[[f'popularity_{i}thMonth' for i in range(1, 13)]].min(axis=1)
    data[f'popularity_Year_max'] = data[[f'popularity_{i}thMonth' for i in range(1, 13)]].max(axis=1)
    data[f'popularity_model_Year_var'] = data[[f'popularity_model_{i}thMonth' for i in range(1,13)]].var(axis=1)
    data[f'popularity_model_Year_mean'] = data[[f'popularity_model_{i}thMonth' for i in range(1,13)]].mean(axis=1)
    data[f'popularity_model_Year_min'] = data[[f'popularity_model_{i}thMonth' for i in range(1,13)]].min(axis=1)
    data[f'popularity_model_Year_max'] = data[[f'popularity_model_{i}thMonth' for i in range(1,13)]].max(axis=1)
    data[f'popularity_pro_Year_var'] = data[[f'popularity_pro_{i}thMonth' for i in range(1, 13)]].var(axis=1)
    data[f'popularity_pro_Year_mean'] = data[[f'popularity_pro_{i}thMonth' for i in range(1, 13)]].mean(axis=1)
    data[f'popularity_pro_Year_min'] = data[[f'popularity_pro_{i}thMonth' for i in range(1, 13)]].min(axis=1)
    data[f'popularity_pro_Year_max'] = data[[f'popularity_pro_{i}thMonth' for i in range(1, 13)]].max(axis=1)
    data[f'popularity_body_Year_var'] = data[[f'popularity_body_{i}thMonth' for i in range(1, 13)]].var(axis=1)
    data[f'popularity_body_Year_mean'] = data[[f'popularity_body_{i}thMonth' for i in range(1, 13)]].mean(axis=1)
    data[f'popularity_body_Year_min'] = data[[f'popularity_body_{i}thMonth' for i in range(1, 13)]].min(axis=1)
    data[f'popularity_body_Year_max'] = data[[f'popularity_body_{i}thMonth' for i in range(1, 13)]].max(axis=1)
    data[f'comment_Year_var'] = data[[f'comment_{i}thMonth' for i in range(1, 13)]].var(axis=1)
    data[f'comment_Year_mean'] = data[[f'comment_{i}thMonth' for i in range(1, 13)]].mean(axis=1)
    data[f'comment_Year_min'] = data[[f'comment_{i}thMonth' for i in range(1, 13)]].min(axis=1)
    data[f'comment_Year_max'] = data[[f'comment_{i}thMonth' for i in range(1, 13)]].max(axis=1)
    data[f'reply_Year_var'] = data[[f'reply_{i}thMonth' for i in range(1,13)]].var(axis=1)
    data[f'reply_Year_mean'] = data[[f'reply_{i}thMonth' for i in range(1,13)]].mean(axis=1)
    data[f'reply_Year_min'] = data[[f'reply_{i}thMonth' for i in range(1,13)]].min(axis=1)
    data[f'reply_Year_max'] = data[[f'reply_{i}thMonth' for i in range(1,13)]].max(axis=1)

    #差、比值特恒
    for i in range(2,12):
        data[f'sales_diff_{i}_{i-1}'] = data[f'sales_{i}thMonth']-data[f'sales_{i-1}thMonth']
        data[f'popularity_diff_{i}_{i-1}'] = data[f'popularity_{i}thMonth']-data[f'popularity_{i-1}thMonth']
        data[f'comment_diff_{i}_{i - 1}'] = data[f'comment_{i}thMonth'] - data[f'comment_{i - 1}thMonth']
        data[f'reply_diff_{i}_{i - 1}'] = data[f'reply_{i}thMonth'] - data[f'reply_{i - 1}thMonth']
        data[f'sales_model_diff_{i}_{i - 1}'] = data[f'reply_{i}thMonth'] - data[f'reply_{i - 1}thMonth']
        data[f'popularity_model_{i}_{i - 1}'] = data[f'reply_{i}thMonth'] - data[f'reply_{i - 1}thMonth']
        data[f'sales_pro_diff_{i}_{i - 1}'] = data[f'reply_{i}thMonth'] - data[f'reply_{i - 1}thMonth']
        data[f'popularity_pro_diff_{i}_{i - 1}'] = data[f'reply_{i}thMonth'] - data[f'reply_{i - 1}thMonth']
        data[f'sales_body_diff_{i}_{i - 1}'] = data[f'reply_{i}thMonth'] - data[f'reply_{i - 1}thMonth']
        data[f'popularity_body_diff_{i}_{i - 1}'] = data[f'reply_{i}thMonth'] - data[f'reply_{i - 1}thMonth']

        data[f'sales_time_{i}_{i-1}'] = data[f'sales_{i}thMonth']/data[f'sales_{i-1}thMonth']
        data[f'popularity_time_{i}_{i-1}'] = data[f'popularity_{i}thMonth']/data[f'popularity_{i-1}thMonth']
        data[f'comment_time_{i}_{i - 1}'] = data[f'comment_{i}thMonth'] / data[f'comment_{i - 1}thMonth']
        data[f'reply_time_{i}_{i - 1}'] = data[f'reply_{i}thMonth'] / data[f'reply_{i - 1}thMonth']
        data[f'sales_model_time_{i}_{i - 1}'] = data[f'reply_{i}thMonth'] / data[f'reply_{i - 1}thMonth']
        data[f'popularity_model_{i}_{i - 1}'] = data[f'reply_{i}thMonth'] / data[f'reply_{i - 1}thMonth']
        data[f'sales_pro_time_{i}_{i - 1}'] = data[f'reply_{i}thMonth'] / data[f'reply_{i - 1}thMonth']
        data[f'popularity_pro_time_{i}_{i - 1}'] = data[f'reply_{i}thMonth'] / data[f'reply_{i - 1}thMonth']
        data[f'sales_body_time_{i}_{i - 1}'] = data[f'reply_{i}thMonth'] / data[f'reply_{i - 1}thMonth']
        data[f'popularity_body_time_{i}_{i - 1}'] = data[f'reply_{i}thMonth'] / data[f'reply_{i - 1}thMonth']

    data['season'] = data['regMonth']%4
    print()
    print(data.info())
    with open(P_DATA + 'train_columns.txt', 'w') as train_f:
        for n in data.columns:
            train_f.write(str(n)+'\n')
    with open(P_DATA + 'train.pk', 'wb') as train_f:
        pickle.dump(data, train_f)

if __name__ == "__main__":
    t1 = time.time()
    feature_main()
    print(time.time()-t1)