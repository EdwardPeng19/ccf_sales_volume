import sys
import numpy as np
import pandas as pd
import os
import gc
from tqdm import tqdm, tqdm_notebook
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import LabelEncoder
import datetime
import time
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
from model import *
from features import *

def eval_model(y_true, y_pre):
    b = np.sum((y_true-y_pre)**2)/len(y_true)
    return np.sqrt(b)/np.mean(y_true)

def get_month_features(data, month):
    sales_features = []
    for fea in ['sales_','sales_model_mean_','sales_province_mean_',
          ]:
        sales_features.extend(fea+f'{th}thMonth' for th in [1,2,3,4,12,])
    popularity_features = []
    for fea in ['popularity_',]:
        popularity_features.extend(fea+f'{th}thMonth' for th in [1,2,3,4])
    comment_features = []
    for fea in ['comment_', ]:
        comment_features.extend(fea + f'{th}thMonth' for th in range(month - 24, 13))
    reply_features = []
    for fea in ['reply_', ]:
        reply_features.extend(fea + f'{th}thMonth' for th in range(month - 24, 13))

    # for fea in ['sales_', 'popularity_', 'comment_', 'reply_',
    #             'sales_province_mean_', 'sales_province_var_', 'popularity_province_mean_', 'popularity_province_var_',
    #             'sales_model_mean_', 'sales_model_var_', 'popularity_model_mean_', 'popularity_model_var_',
    #             'sales_body_mean_', 'sales_body_var_', 'popularity_body_mean_', 'popularity_body_var_']:
    #     features.extend(fea+f'{th}thMonth' for th in range(1,13))
    #窗口特征
    window_features = []
    for win_size in [3,4,5,6,7,12]:
        for fea in ['sales_','sales_model_mean_','sales_model_var_', 'comment_', 'reply_', ]:
            window_features.extend([ fea+f'window_mean_{win_size}',])
    #一阶趋势
    trend_features = []
    for i in [1, 2, 3, 4]:
        j = i+1
        for fea in ['sales_',]:
            trend_features.append(fea+f'_diff_{i}_{j}')
            trend_features.append(fea+f'_time_{i}_{j}')
    #二阶趋势
    trend_features2 = []
    for i in [1, 2, 3, 4]:
        j = i+1
        k = i+2
        for fea in ['sales_',]:
            trend_features2.append(fea+f'_diff_{i}_{j}2')
            #trend_features2.append(fea+f'_time_{i}_{j}2')
    sales_thQuarter = []
    for sea in [1, 2, 3, 4]:
        sales_thQuarter.extend([f'sales_{sea}thQuarter_var', f'sales_pro_{sea}thQuarter_var', ])
    if month == 25:
        features = sales_features+window_features+trend_features+trend_features2
        '''
        test_score:0.7964846255854958
        test_score:0.7965158865570552
        '''
    elif month == 26:
        sales_features = []
        for fea in ['sales_', 'sales_model_mean_', 'sales_province_mean_',
                    ]:
            sales_features.extend(fea + f'{th}thMonth' for th in [2, 3, 4, 5, 12 ])
        popularity_features = []
        for fea in ['popularity_', ]:
            popularity_features.extend(fea + f'{th}thMonth' for th in [2, 3, 4])
        comment_features = []
        for fea in ['comment_', ]:
            comment_features.extend(fea + f'{th}thMonth' for th in range(month - 24, 13))
        reply_features = []
        for fea in ['reply_', ]:
            reply_features.extend(fea + f'{th}thMonth' for th in range(month - 24, 13))

        # for fea in ['sales_', 'popularity_', 'comment_', 'reply_',
        #             'sales_province_mean_', 'sales_province_var_', 'popularity_province_mean_', 'popularity_province_var_',
        #             'sales_model_mean_', 'sales_model_var_', 'popularity_model_mean_', 'popularity_model_var_',
        #             'sales_body_mean_', 'sales_body_var_', 'popularity_body_mean_', 'popularity_body_var_']:
        #     features.extend(fea+f'{th}thMonth' for th in range(1,13))
        # 窗口特征
        window_features = []
        for win_size in [3, 4, 5, 6, 7]:
            for fea in ['sales_', 'sales_model_mean_',]:
                window_features.extend([fea + f'window2_mean_{win_size}'])
        # 一阶趋势
        trend_features = []
        for i in [2, 3, 4]:
            j = i + 1
            for fea in ['sales_', 'sales_model_mean_',]:
                trend_features.append(fea + f'_diff_{i}_{j}')
                trend_features.append(fea + f'_time_{i}_{j}')
        # 二阶趋势
        trend_features2 = []
        for i in [2, 3]:
            j = i + 1
            k = i + 2
            for fea in ['sales_', 'sales_model_mean_']:
                trend_features2.append(fea + f'_diff_{i}_{j}2')
                trend_features2.append(fea + f'_time_{i}_{j}2')
        sales_thQuarter = []
        for sea in [2,4]:
            sales_thQuarter.extend([f'sales_{sea}thQuarter_var', f'sales_pro_{sea}thQuarter_var', ])
        features = sales_features + window_features
        '''
        test_score:0.7213469639120724
        test_score:0.7230700387215288
        '''
    elif month == 27:
        sales_features = []
        for fea in ['sales_', 'sales_model_mean_','sales_model_var_'
                    ]:
            sales_features.extend(fea + f'{th}thMonth' for th in [1,2,3, 4, 5, 6, ])
        popularity_features = []
        for fea in ['popularity_', ]:
            popularity_features.extend(fea + f'{th}thMonth' for th in [2, 3, 4])
        comment_features = []
        for fea in ['comment_', ]:
            comment_features.extend(fea + f'{th}thMonth' for th in range(month - 24, 13))
        reply_features = []
        for fea in ['reply_', ]:
            reply_features.extend(fea + f'{th}thMonth' for th in range(month - 24, 13))

        # for fea in ['sales_', 'popularity_', 'comment_', 'reply_',
        #             'sales_province_mean_', 'sales_province_var_', 'popularity_province_mean_', 'popularity_province_var_',
        #             'sales_model_mean_', 'sales_model_var_', 'popularity_model_mean_', 'popularity_model_var_',
        #             'sales_body_mean_', 'sales_body_var_', 'popularity_body_mean_', 'popularity_body_var_']:
        #     features.extend(fea+f'{th}thMonth' for th in range(1,13))
        # 窗口特征
        window_features = []
        for win_size in [3, 4, 5, 6, 7, 12]:
            for fea in ['sales_', 'sales_model_mean_',
                        ]:
                window_features.extend([fea + f'window3_mean_{win_size}'])
        # 一阶趋势
        trend_features = []
        for i in [3, 4]:
            j = i + 1
            for fea in ['sales_', 'sales_model_mean_', ]:
                trend_features.append(fea + f'_diff_{i}_{j}')
                trend_features.append(fea + f'_time_{i}_{j}')
        # 二阶趋势
        trend_features2 = []
        for i in [2, 3]:
            j = i + 1
            k = i + 2
            for fea in ['sales_', 'sales_model_mean_']:
                trend_features2.append(fea + f'_diff_{i}_{j}2')
                trend_features2.append(fea + f'_time_{i}_{j}2')
        sales_thQuarter = []
        for sea in [2]:
            sales_thQuarter.extend([f'sales_{sea}thQuarter_var', f'sales_pro_{sea}thQuarter_var', ])
        features = sales_features + trend_features + window_features + trend_features2
    else:
        sales_features = []
        for fea in ['sales_', 'sales_model_mean_', 'sales_model_var_'
                    ]:
            sales_features.extend(fea + f'{th}thMonth' for th in [4,5,6,7,8,9,12 ])
        popularity_features = []
        for fea in ['popularity_', ]:
            popularity_features.extend(fea + f'{th}thMonth' for th in [2, 3, 4])
        comment_features = []
        for fea in ['comment_', ]:
            comment_features.extend(fea + f'{th}thMonth' for th in range(month - 24, 13))
        reply_features = []
        for fea in ['reply_', ]:
            reply_features.extend(fea + f'{th}thMonth' for th in range(month - 24, 13))

        # for fea in ['sales_', 'popularity_', 'comment_', 'reply_',
        #             'sales_province_mean_', 'sales_province_var_', 'popularity_province_mean_', 'popularity_province_var_',
        #             'sales_model_mean_', 'sales_model_var_', 'popularity_model_mean_', 'popularity_model_var_',
        #             'sales_body_mean_', 'sales_body_var_', 'popularity_body_mean_', 'popularity_body_var_']:
        #     features.extend(fea+f'{th}thMonth' for th in range(1,13))
        # 窗口特征
        window_features = []
        for win_size in [3, 4, 5, 6, 7, 12]:
            for fea in ['sales_', 'sales_model_mean_',
                        ]:
                window_features.extend([fea + f'window4_mean_{win_size}'])
        # 一阶趋势
        trend_features = []
        for i in [3, 4]:
            j = i + 1
            for fea in ['sales_', 'sales_model_mean_', ]:
                trend_features.append(fea + f'_diff_{i}_{j}')
                trend_features.append(fea + f'_time_{i}_{j}')
        # 二阶趋势
        trend_features2 = []
        for i in [2, 3]:
            j = i + 1
            k = i + 2
            for fea in ['sales_', 'sales_model_mean_']:
                trend_features2.append(fea + f'_diff_{i}_{j}2')
                trend_features2.append(fea + f'_time_{i}_{j}2')
        sales_thQuarter = []
        for sea in [2]:
            sales_thQuarter.extend([f'sales_{sea}thQuarter_var', f'sales_pro_{sea}thQuarter_var', ])
        features = sales_features
    '''
    test_score:0.6093057297275858
    test_score:0.6053310504730571
    '''
    print(f'特征数量为{len(features)}')
    return data, features

def main(month, ym_cut, offline):

    data = pd.read_pickle(P_DATA + 'train.pk')
    #va_data = pd.read_pickle(P_DATA + 'vali.pk')
    model_type = 'lgb'
    label = 'label'
    data['season'] = data['regMonth'] % 4
    # carCommentVolum newsReplyVolum popularity
    data, features = get_month_features(data, month)
    # 切分训练集、测试集
    test_m = data[(data['ym'] == month)]
    train_m = data[(data['ym'] < month) & (data['ym'] >= ym_cut)]
    numerical_features = ['adcode', 'regYear', 'regMonth',
                          ] + features
    category_features1 = ['province', 'model', 'bodyType', 'mp_fea',
                          ]  # 需转换为数值类型
    category_features2 = []
    print(data.shape)
    split_info = (
        'ym',
        [ym_cut,month-5],
        [month-4, month-4]
    )
    best_iter = 500
    if month == 25:
        best_iter = 428
    elif month == 26:
        best_iter = 607
    elif month == 27:
        best_iter = 149
    elif month == 28:
        best_iter = 228
    #model, valid_df, test_pre = reg_model_v2(train_m, test_m, label, model_type, numerical_features, category_features1,category_features2, split_info, best_iter)
    model, valid_df, test_pre = reg_model(train_m, test_m, label, model_type, numerical_features, category_features1,category_features2, 2019)

    valid_df['y_hat'] = valid_df['y_hat'].apply(lambda x: 0 if x < 0 else x)
    valid_df = valid_df.round().astype(int)

    #models = valid_df['model'].value_counts().index
    mEvalTest = 0
    for m in range(0,60):
        # 测试机NRMSE
        df_test = valid_df[valid_df['model'] == m]
        m_eval_test = eval_model(df_test['y'], df_test['y_hat'])
        mEvalTest = mEvalTest + m_eval_test
    test_score = 1 - mEvalTest / 60
    print(f'test_score:{test_score}')

    # #预测销量
    test_m['forecastVolum'] = test_pre
    test_m['forecastVolum'] = test_m['forecastVolum'].apply(lambda x: 0 if x < 0 else x)
    test = test_m[['id', 'forecastVolum',]].round().astype(int)

    if offline:
        return valid_df, test_score
    else:
        print(np.mean(test['forecastVolum']))
        return test, test_score

if __name__ == "__main__":
    test_DF = pd.DataFrame()
    offline = True
    #预测第一个月
    feature_main()
    total_loss = 0
    for month,ym_cut in zip([25,26,27,28], [13,13,13,13]):
        label_df, score = main(month, ym_cut, offline)
        #sys.exit(-1)
        test_DF = test_DF.append(label_df)
        test_DF.to_csv(P_SUBMIT + f'ccf_car_label_lgb_{month}month.csv', index=False)
        feature_main(month+1, P_SUBMIT + f'ccf_car_label_lgb_{month}month.csv', offline)
        total_loss = total_loss + score

    #rule = pd.read_csv('./Baseline/sub_rule.csv').sort_values(by='id').reset_index()
    print(total_loss/4)
    '''
    0.6980048968569194
    0.707372805380126
    '''
    #sys.exit(-1)
    test_DF.to_csv(P_SUBMIT + f'ccf_car_sales_lgb_full.csv', index=False)
    print('-------------------------------------------------')
    bb = pd.read_csv(P_SUBMIT + 'ccf_car_sales_lgb_full.csv', ).sort_values(by='id').reset_index(drop=True)
    print(np.mean(bb['forecastVolum']))
    print(np.sum(bb['forecastVolum']))
    print(np.mean(bb[bb['id'] <= 1342]['forecastVolum']))
    print(np.mean(bb[(bb['id'] > 1342) & (bb['id'] <= 2684)]['forecastVolum']))
    print(np.mean(bb[(bb['id'] > 2684) & (bb['id'] <= 4026)]['forecastVolum']))
    print(np.mean(bb[(bb['id'] > 4026)]['forecastVolum']))

'''
0.7464221765836813
436.7695075757576
2306143
497.6901515151515
316.1068181818182
468.6628787878788
464.6181818181818

0.730972614900746
'''

