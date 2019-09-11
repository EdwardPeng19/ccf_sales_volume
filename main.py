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
    for sea in [1,4]:
        sales_thQuarter.extend([f'sales_{sea}thQuarter_var', f'sales_model_{sea}thQuarter_var',
                                ])

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
    features = sales_thMonth + sales_model_thMonth + sales_pro_thMonth + \
               sales_thQuarter
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

        season_features = []
        for sea in [1, 2, 3, 4]:
            season_features.extend([f'sales_allp_{sea}thSeason_mean', f'sales_allp_{sea}thSeason_var'])
        features = sales_thMonth + sales_model_thMonth+sales_body_thMonth + comment_thMonth + reply_thMonth + year_features

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
    return data, features


def main(month, offline):
    data = pd.read_pickle(P_DATA + 'train.pk')
    model_type = 'lgb'
    label = 'label'
    # carCommentVolum newsReplyVolum popularity
    data, features = get_month_features(data, month)
    # 切分训练集、测试集
    train_m = data[(data['regYear'] == 2017) | ( (data['regYear'] == 2018) & (data['regMonth'] <= month-1) )]
    test_m = data[(data['regYear'] == 2018) & (data['regMonth'] == month)]
    if offline:
        test_m = train_m[(train_m['regYear'] == 2017) & (train_m['regMonth'] >= 12)]
        train_m = train_m[(train_m['regYear'] != 2017) | (train_m['regMonth'] < 12)]

    numerical_features = ['adcode', 'regYear', 'regMonth',
                          ] + features
    category_features1 = ['province', 'model', 'bodyType', 'mp_fea','season'
                          ]  # 需转换为数值类型
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
    #预测第一个月
    #feature_main()
    label_df = main(1, False)
    print(np.mean(label_df['forecastVolum']))
    test_DF = test_DF.append(label_df)
    test_DF.to_csv(P_SUBMIT + f'ccf_car_label_lgb_1month.csv', index=False)

    # 预测第二个月
    feature_main(P_SUBMIT + f'ccf_car_label_lgb_1month.csv')
    label_df = main(2, False)
    print(np.mean(label_df['forecastVolum']))
    test_DF = test_DF.append(label_df)
    test_DF.to_csv(P_SUBMIT + f'ccf_car_label_lgb_2month.csv', index=False)

    #预测第三个月
    feature_main(P_SUBMIT + f'ccf_car_label_lgb_2month.csv')
    label_df = main(3, False)
    print(np.mean(label_df['forecastVolum']))
    test_DF = test_DF.append(label_df)
    test_DF.to_csv(P_SUBMIT + f'ccf_car_label_lgb_3month.csv', index=False)

    # 预测第四个月
    feature_main(P_SUBMIT + f'ccf_car_label_lgb_3month.csv')
    label_df = main(4, False)
    print(np.mean(label_df['forecastVolum']))
    test_DF = test_DF.append(label_df)
    test_DF.to_csv(P_SUBMIT + f'ccf_car_sales_lgb_full.csv', index=False)




'''
test_mae:185.83409090909092,mse:145611.7553030303,score:0.6808523989600382, train_mae:66.99800275482093,mse:18982.71108815427, score:0.7725821409538414
test_mae:183.65454545454546,mse:145163.44545454546,score:0.6848456123728662, train_mae:67.04724517906337,mse:19187.598898071625, score:0.7729141629121636
test_mae:191.30530303030304,mse:156012.3143939394,score:0.6877802680196621, train_mae:68.63601928374656,mse:20682.567424242425, score:0.7697593573728629
test_mae:189.2651515151515,mse:153034.5303030303,score:0.690085440369921, train_mae:67.95798898071625,mse:19814.765013774104, score:0.770959874606481

'''