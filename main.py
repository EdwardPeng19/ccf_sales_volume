import numpy as np
import pandas as pd
import os
import sys
import gc
from tqdm import tqdm, tqdm_notebook
import datetime
import time
from sklearn.metrics import f1_score,mean_absolute_error,mean_squared_error
from model import *
from features import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

def eval_model(y_true, y_pre):
    b = np.sum((y_true-y_pre)**2)/len(y_true)
    return np.sqrt(b)/np.mean(y_true)

features = []
features1 = []
sales_thMonth = [f'sales_{th - 1}thMonth' for th in range(2, 14)]



def get_month_features(data, month):
    cat_features = []
    sales_thMonth = [f'sales_{th}thMonth' for th in [1,2,3,4,12]]
    sales_model_thMonth = [f'sales_model_{th}thMonth' for th in [1,2,3,4,12]]
    sales_pro_thMonth = [f'sales_pro_{th}thMonth' for th in [1,2,3,4,12]]
    sales_body_thMonth = [f'sales_body_{th}thMonth' for th in list(range(1,month+4))+[12]]  # 效果不好

    popularity_thMonth = [f'popularity_{th}thMonth' for th in range(month, 13)]
    popularity_model_thMonth = [f'popularity_model_{th}thMonth' for th in range(month, 13)]
    popularity_pro_thMonth = [f'popularity_pro_{th}thMonth' for th in range(1, 13)]  # 效果不好
    popularity_body_thMonth = [f'popularity_body_{th}thMonth' for th in range(1, 13)]  # 效果不好
    comment_thMonth = [f'comment_{th}thMonth' for th in range(12, 13)]  # 效果不好
    reply_thMonth = [f'reply_{th}thMonth' for th in range(12, 13)]  # 效果不好

    sales_thQuarter = []
    for sea in [1, 4]:
        sales_thQuarter.extend([f'sales_{sea}thQuarter_var', f'sales_model_{sea}thQuarter_var',])

    # 差、比值特恒
    diff_features = []
    time_features = []
    for i in range(2, 4):
        diff_features.append(f'sales_diff_{i}_{i - 1}')
        time_features.append(f'sales_time_{i}_{i - 1}')
    year_features = [f'sales_Year_var', f'sales_Year_mean', f'sales_Year_min', f'sales_Year_max',
                     'sales_model_Year_var', 'sales_model_Year_mean', 'sales_model_Year_min', 'sales_model_Year_max',
                     'sales_pro_Year_var', 'sales_pro_Year_mean', 'sales_pro_Year_min', 'sales_pro_Year_mean',
                     'sales_body_Year_var', 'sales_body_Year_mean', 'sales_body_Year_min', 'sales_body_Year_max']
    features = sales_thMonth + sales_model_thMonth + sales_pro_thMonth + popularity_thMonth + popularity_model_thMonth + comment_thMonth + reply_thMonth + \
               sales_thQuarter + diff_features+time_features
    if month == 1:
        features = features
    elif month == 2:
        features =  features
    elif month == 3:
        sales_thMonth = [f'sales_{th}thMonth' for th in range(1,12)]
        sales_model_thMonth = [f'sales_model_{th}thMonth' for th in range(month,12)]
        sales_pro_thMonth = [f'sales_pro_{th}thMonth' for th in range(1,12)]
        sales_body_thMonth = [f'sales_body_{th}thMonth' for th in range(month,12)]  # 效果不好

        popularity_thMonth = [f'popularity_{th}thMonth' for th in range(month, 13)]
        popularity_model_thMonth = [f'popularity_model_{th}thMonth' for th in range(month, 13)]
        popularity_pro_thMonth = [f'popularity_pro_{th}thMonth' for th in range(1, 13)]  # 效果不好
        popularity_body_thMonth = [f'popularity_body_{th}thMonth' for th in range(1, 13)]  # 效果不好
        comment_thMonth = [f'comment_{th}thMonth' for th in range(month, 13)]  # 效果不好
        reply_thMonth = [f'reply_{th}thMonth' for th in range(month, 13)]  # 效果不好

        sales_thQuarter = []
        # for sea in [1, 2, 3, 4]:
        #     tmp = [f'sales_{sea}thQuarter_var',
        #            f'sales_model_{sea}thQuarter_var',
        #            ]
        #     sales_thQuarter.extend(tmp)
        for hy in [1, 2]:
            tmp = [f'sales_{hy}thHalfyear_var', f'sales_{hy}thHalfyear_mean',
                   f'sales_model_{hy}thHalfyear_var', f'sales_model_{hy}thHalfyear_mean', f'sales_model_{hy}thHalfyear_min', f'sales_model_{hy}thHalfyear_max',
                   f'sales_pro_{hy}thHalfyear_var', f'sales_pro_{hy}thHalfyear_mean', f'sales_pro_{hy}thHalfyear_min', f'sales_pro_{hy}thHalfyear_max',
                   f'sales_body_{hy}thHalfyear_var', f'sales_body_{hy}thHalfyear_mean', f'sales_body_{hy}thHalfyear_min', f'sales_body_{hy}thHalfyear_max',
                   ]
            sales_thQuarter.extend(tmp)
        year_features = [f'sales_Year_var', f'sales_Year_mean', f'sales_Year_min', f'sales_Year_max',
                         'sales_model_Year_var', 'sales_model_Year_mean', 'sales_model_Year_min', 'sales_model_Year_max',
                         'sales_pro_Year_var', 'sales_pro_Year_mean', 'sales_pro_Year_min', 'sales_pro_Year_mean',
                         'sales_body_Year_var', 'sales_body_Year_mean', 'sales_body_Year_min', 'sales_body_Year_max']
        # 差、比值特恒
        diff_features = []
        time_features = []
        for i in range(2, 4):
            diff_features.append(f'sales_diff_{i}_{i - 1}')
            diff_features.append(f'sales_model_diff_{i}_{i - 1}')
            #diff_features.append(f'sales_pro_diff_{i}_{i - 1}')
            #diff_features.append(f'sales_body_diff_{i}_{i - 1}')


            time_features.append(f'sales_time_{i}_{i - 1}')
            time_features.append(f'sales_model_time_{i}_{i - 1}')
            #time_features.append(f'sales_pro_time_{i}_{i - 1}')
            #time_features.append(f'sales_body_time_{i}_{i - 1}')


        features = sales_thMonth + sales_model_thMonth+sales_body_thMonth + comment_thMonth + reply_thMonth + year_features
        #cat_features.extend(['pb', 'pm', 'bm', 'mm'])
        ''''

464.2462121212121
461.4613636363636


        '''
        #features = features
    elif month == 4:
        sales_thMonth = [f'sales_{th}thMonth' for th in range(1, 12)]
        sales_model_thMonth = [f'sales_model_{th}thMonth' for th in range(month, 12)]
        sales_pro_thMonth = [f'sales_pro_{th}thMonth' for th in range(1, 12)]
        sales_body_thMonth = [f'sales_body_{th}thMonth' for th in range(month, 12)]  # 效果不好

        popularity_thMonth = [f'popularity_{th}thMonth' for th in range(month, 13)]
        popularity_model_thMonth = [f'popularity_model_{th}thMonth' for th in range(month, 13)]
        popularity_pro_thMonth = [f'popularity_pro_{th}thMonth' for th in range(1, 13)]  # 效果不好
        popularity_body_thMonth = [f'popularity_body_{th}thMonth' for th in range(1, 13)]  # 效果不好
        comment_thMonth = [f'comment_{th}thMonth' for th in range(month, 13)]  # 效果不好
        reply_thMonth = [f'reply_{th}thMonth' for th in range(month, 13)]  # 效果不好

        sales_thQuarter = []
        # for sea in [1, 2, 3, 4]:
        #     tmp = [f'sales_{sea}thQuarter_var', f'sales_{sea}thQuarter_mean', f'sales_{sea}thQuarter_min', f'sales_{sea}thQuarter_max',
        #            f'sales_model_{sea}thQuarter_var', f'sales_model_{sea}thQuarter_mean', f'sales_model_{sea}thQuarter_min', f'sales_model_{sea}thQuarter_max',
        #            ]
        #     sales_thQuarter.extend(tmp)
        # for hy in [1, 2]:
        #     tmp = [f'sales_{hy}thHalfyear_var', f'sales_{hy}thHalfyear_mean', f'sales_{hy}thHalfyear_min', f'sales_{hy}thHalfyear_max']
        #     sales_thQuarter.extend(tmp)
        # 差、比值特恒
        diff_features = []
        time_features = []
        for i in range(2, 4):
            diff_features.append(f'sales_diff_{i}_{i - 1}')
            time_features.append(f'sales_time_{i}_{i - 1}')
        features = sales_thMonth + sales_model_thMonth + sales_body_thMonth + comment_thMonth + reply_thMonth
    else:
        print("输入月份不合法")
        features = []
    print(f'特征数量为{len(features)}')
    return data, features, cat_features


def main(month, offline):
    data = pd.read_pickle(P_DATA + 'train.pk')
    model_type = 'lgb'
    label = 'label'
    # carCommentVolum newsReplyVolum popularity
    data, features, cat_features = get_month_features(data, month)
    data['pb'] = data['province'] + data['bodyType']
    data['pm'] = data['regMonth'].astype(str) + data['province']
    data['bm'] = data['regMonth'].astype(str) + data['bodyType']
    data['mm'] = data['regMonth'].astype(str) + data['model']
    data['season'] = data['regMonth'] % 4

    # 切分训练集、测试集
    train_m = data[(data['regYear'] == 2017) | ( (data['regYear'] == 2018) & (data['regMonth'] <= month-1) )]
    test_m = data[(data['regYear'] == 2018) & (data['regMonth'] == month)]
    if offline:
        test_m = train_m[(train_m['regYear'] == 2017) & (train_m['regMonth'] >= 12)]
        train_m = train_m[(train_m['regYear'] != 2017) | (train_m['regMonth'] < 12)]

    numerical_features = ['adcode', 'regYear', 'regMonth',
                          'province_sales_mean', 'mean_model_salesVolume', 'mean_bodyType_salesVolume', 'mean_province_popularity', 'mean_model_popularity', 'mean_model_newsReplyVolum', 'mean_model_carCommentVolum',
                          'mean_province_permonth_salesVolume',
                          'mean_model_permonth_salesVolume',
                          'mean_bodyType_permonth_salesVolume',
                          'mean_province_permonth_popularity',
                          'mean_model_permonth_popularity',
                          'mean_model_permonth_newsReplyVolum',
                          'mean_model_permonth_carCommentVolum',
                          'season_sales_mean','season_sales_var'
                          ] + features
    category_features1 = ['province', 'model', 'bodyType', 'mp_fea',

                          ] + cat_features + ['season']  # 需转换为数值类型
    category_features2 = []

    pred, train_pred = reg_model(train_m, test_m, label, model_type, numerical_features, category_features1,category_features2, seed=2019)
    test_m['forecastVolum'] = pred
    test_m['forecastVolum'] = test_m['forecastVolum'].apply(lambda x: 0 if x < 0 else x)
    train_m['forecastVolum'] = train_pred
    train_m['forecastVolum'] = train_m['forecastVolum'].apply(lambda x: 0 if x < 0 else x)
    train_m[['id', 'forecastVolum', 'label']] = train_m[['id', 'forecastVolum', 'label']].round().astype(int)
    if offline:
        # 线下评测
        test_m[['id', 'forecastVolum', 'label']] = test_m[['id', 'forecastVolum', 'label']].round().astype(int)
        mae_train = mean_absolute_error(train_m['label'], train_m['forecastVolum'])
        mse_train = mean_squared_error(train_m['label'], train_m['forecastVolum'])
        mae = mean_absolute_error(test_m['label'], test_m['forecastVolum'])
        mse = mean_squared_error(test_m['label'], test_m['forecastVolum'])

        models = data['model'].value_counts().index
        mEvalTrain = 0
        mEvalTest = 0
        for m in models:
            # 训练集NRMSE
            df_train = train_m[train_m['model'] == m]
            m_eval_train = eval_model(df_train['label'], df_train['forecastVolum'])
            mEvalTrain = mEvalTrain + m_eval_train
            # 测试机NRMSE
            df_test = test_m[test_m['model'] == m]
            m_eval_test = eval_model(df_test['label'], df_test['forecastVolum'])
            mEvalTest = mEvalTest + m_eval_test
        train_score = 1 - mEvalTrain / len(models)
        test_score = 1 - mEvalTest / len(models)

        print(f'test_mae:{mae},mse:{mse},score:{test_score}, train_mae:{mae_train},mse:{mse_train}, score:{train_score}')
        sys.exit(-1)
    else:
        # #预测销量
        mae_train = mean_absolute_error(train_m['label'], train_m['forecastVolum'])
        mse_train = mean_squared_error(train_m['label'], train_m['forecastVolum'])
        print(f'train_mae:{mae_train},train_mse:{mse_train}')
        test = test_m[['id', 'forecastVolum']].round().astype(int)
        #test.to_csv(P_SUBMIT + f'ccf_car_sales_lgb_{today}.csv', index=False)
        return test

if __name__ == "__main__":
    #分别运行4次 预测1月 2月 3月 4月
    test_DF = pd.DataFrame()

    for month in [1,2,3,4]:
        if month > 1:
            feature_main(P_SUBMIT + f'ccf_car_label_lgb_{month-1}month.csv')

        label_df = main(month, False)
        print(np.mean(label_df['forecastVolum']))
        test_DF = test_DF.append(label_df)
        test_DF.to_csv(P_SUBMIT + f'ccf_car_label_lgb_{month}month.csv', index=False)


    test_DF.to_csv(P_SUBMIT + f'ccf_car_sales_lgb_full.csv', index=False)


'''
test_mae:180.12272727272727,mse:140846.29393939395,score:0.6919838231395913, train_mae:59.80075757575757,mse:16059.620041322314, score:0.7972470712372305
test_mae:179.6780303030303,mse:139983.44924242425,score:0.6911457278111421, train_mae:59.9784435261708,mse:16268.819490358126, score:0.7971942177221922

test_mae:169.8719696969697,mse:129632.43106060606,score:0.7016937797994294, train_mae:58.89001377410468,mse:15800.620730027547, score:0.8016588206221924

'''