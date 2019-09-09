# -*- coding: utf-8 -*-
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

path = "../data/"

# 训练数据集
# 历史销量数据
train_sales_data = pd.read_csv(path + "train_sales_data.csv")
# 车型搜索数据
train_search_data = pd.read_csv(path + 'train_search_data.csv')
# 汽车垂直媒体新闻评论数据和车型评论数据
train_user_reply_data = pd.read_csv(path + 'train_user_reply_data.csv')
# 测试数据集
test_sales_data = pd.read_csv(path + 'evaluation_public.csv')

#  **********2018年1月，提取方式:历史月份销量比例，考虑时间衰减，月份越近占比越高**********
# 季度
# 各个省份所有车型：2017年1月的销量 / 2016年12月份的销量  1320个比例值
m1_12 = train_sales_data.loc[(train_sales_data.regYear==2017)&(train_sales_data.regMonth==1) , 'salesVolume'].values / train_sales_data.loc[(train_sales_data.regYear==2016)&(train_sales_data.regMonth==12), 'salesVolume'].values
# 各个省份所有车型：2017年1月的销量 / 2016年11月份的销量  1320个比例值
m1_11 = train_sales_data.loc[(train_sales_data.regYear==2017)&(train_sales_data.regMonth==1) , 'salesVolume'].values / train_sales_data.loc[(train_sales_data.regYear==2016)&(train_sales_data.regMonth==11), 'salesVolume'].values
# 各个省份所有车型：2017年1月的销量 / 2016年10月份的销量  1320个比例值
m1_10 = train_sales_data.loc[(train_sales_data.regYear==2017)&(train_sales_data.regMonth==1) , 'salesVolume'].values / train_sales_data.loc[(train_sales_data.regYear==2016)&(train_sales_data.regMonth==10), 'salesVolume'].values
# 各个省份所有车型：2017年1月的销量 / 2016年9月份的销量  1320个比例值
m1_09 = train_sales_data.loc[(train_sales_data.regYear==2017)&(train_sales_data.regMonth==1) , 'salesVolume'].values / train_sales_data.loc[(train_sales_data.regYear==2016)&(train_sales_data.regMonth==9) , 'salesVolume'].values
# 年
# 各个省份所有车型：2017年1月的销量 / 2016年1月份的销量  1320个比例值
m1_001 = train_sales_data.loc[(train_sales_data.regYear==2017)&(train_sales_data.regMonth==1) , 'salesVolume'].values / train_sales_data.loc[(train_sales_data.regYear==2016)&(train_sales_data.regMonth==1) , 'salesVolume'].values

# 再用2017年9月-12月份的值分别乘以刚才的比例得到2018年1月份的销量
m1_12_volum = train_sales_data.loc[(train_sales_data.regYear==2017)&(train_sales_data.regMonth==12), 'salesVolume'].values * m1_12
m1_11_volum = train_sales_data.loc[(train_sales_data.regYear==2017)&(train_sales_data.regMonth==11), 'salesVolume'].values * m1_11
m1_10_volum = train_sales_data.loc[(train_sales_data.regYear==2017)&(train_sales_data.regMonth==10), 'salesVolume'].values * m1_10
m1_09_volum = train_sales_data.loc[(train_sales_data.regYear==2017)&(train_sales_data.regMonth==9) , 'salesVolume'].values * m1_09
m1_001_volum = train_sales_data.loc[(train_sales_data.regYear==2017)&(train_sales_data.regMonth==8) , 'salesVolume'].values * m1_001

# 2017年9月-12月份对于2018年1月的预测：贡献比列逐渐在减少
test_sales_data.loc[test_sales_data.regMonth==1, 'forecastVolum'] =  m1_12_volum/2 + m1_11_volum/4 + m1_10_volum/8 + m1_09_volum/16 + m1_001_volum/16


#  **********2018年2月，提取方式:历史月份销量比例，考虑时间衰减，月份越近占比越高**********
m2_01 = train_sales_data.loc[(train_sales_data.regYear==2017)&(train_sales_data.regMonth==2) , 'salesVolume'].values / train_sales_data.loc[(train_sales_data.regYear==2017)&(train_sales_data.regMonth==1), 'salesVolume'].values
m2_12 = train_sales_data.loc[(train_sales_data.regYear==2017)&(train_sales_data.regMonth==2) , 'salesVolume'].values / train_sales_data.loc[(train_sales_data.regYear==2016)&(train_sales_data.regMonth==12), 'salesVolume'].values
m2_11 = train_sales_data.loc[(train_sales_data.regYear==2017)&(train_sales_data.regMonth==2) , 'salesVolume'].values / train_sales_data.loc[(train_sales_data.regYear==2016)&(train_sales_data.regMonth==11), 'salesVolume'].values
m2_10 = train_sales_data.loc[(train_sales_data.regYear==2017)&(train_sales_data.regMonth==2) , 'salesVolume'].values / train_sales_data.loc[(train_sales_data.regYear==2016)&(train_sales_data.regMonth==10) , 'salesVolume'].values

m2_002 = train_sales_data.loc[(train_sales_data.regYear==2017)&(train_sales_data.regMonth==2) , 'salesVolume'].values / train_sales_data.loc[(train_sales_data.regYear==2016)&(train_sales_data.regMonth==2) , 'salesVolume'].values

m2_01_volum = test_sales_data.loc[(test_sales_data.regYear==2018)&(test_sales_data.regMonth==1),  'forecastVolum'].values * m2_01
m2_12_volum = train_sales_data.loc[(train_sales_data.regYear==2017)&(train_sales_data.regMonth==12), 'salesVolume'].values * m2_12
m2_11_volum = train_sales_data.loc[(train_sales_data.regYear==2017)&(train_sales_data.regMonth==11), 'salesVolume'].values * m2_11
m2_10_volum = train_sales_data.loc[(train_sales_data.regYear==2017)&(train_sales_data.regMonth==10) , 'salesVolume'].values * m2_10
m2_002_volum = train_sales_data.loc[(train_sales_data.regYear==2017)&(train_sales_data.regMonth==2) , 'salesVolume'].values * m2_002

test_sales_data.loc[test_sales_data.regMonth==2, 'forecastVolum'] =  m2_01_volum/2 + m2_12_volum/4 + m2_11_volum/8 + m2_10_volum/16 + m2_002_volum/16


#  **********2018年3月，提取方式:历史月份销量比例，考虑时间衰减，月份越近占比越高**********
m3_02 = train_sales_data.loc[(train_sales_data.regYear==2017)&(train_sales_data.regMonth==3) , 'salesVolume'].values / train_sales_data.loc[(train_sales_data.regYear==2017)&(train_sales_data.regMonth==2), 'salesVolume'].values
m3_01 = train_sales_data.loc[(train_sales_data.regYear==2017)&(train_sales_data.regMonth==3) , 'salesVolume'].values / train_sales_data.loc[(train_sales_data.regYear==2017)&(train_sales_data.regMonth==1), 'salesVolume'].values
m3_12 = train_sales_data.loc[(train_sales_data.regYear==2017)&(train_sales_data.regMonth==3) , 'salesVolume'].values / train_sales_data.loc[(train_sales_data.regYear==2016)&(train_sales_data.regMonth==12), 'salesVolume'].values
m3_11 = train_sales_data.loc[(train_sales_data.regYear==2017)&(train_sales_data.regMonth==3) , 'salesVolume'].values / train_sales_data.loc[(train_sales_data.regYear==2016)&(train_sales_data.regMonth==11) , 'salesVolume'].values

m3_003 = train_sales_data.loc[(train_sales_data.regYear==2017)&(train_sales_data.regMonth==3) , 'salesVolume'].values / train_sales_data.loc[(train_sales_data.regYear==2016)&(train_sales_data.regMonth==3) , 'salesVolume'].values

m3_02_volum = test_sales_data.loc[(test_sales_data.regYear==2018)&(test_sales_data.regMonth==2),  'forecastVolum'].values * m3_02
m3_01_volum =  test_sales_data.loc[( test_sales_data.regYear==2018)&( test_sales_data.regMonth==1), 'forecastVolum'].values * m3_01
m3_12_volum = train_sales_data.loc[(train_sales_data.regYear==2017)&(train_sales_data.regMonth==12), 'salesVolume'].values * m3_12
m3_11_volum = train_sales_data.loc[(train_sales_data.regYear==2017)&(train_sales_data.regMonth==11) , 'salesVolume'].values * m3_11
m3_003_volum = train_sales_data.loc[(train_sales_data.regYear==2017)&(train_sales_data.regMonth==3) , 'salesVolume'].values * m3_003

test_sales_data.loc[test_sales_data.regMonth==3, 'forecastVolum'] =  m3_02_volum/2 + m3_01_volum/4 + m3_12_volum/8 + m3_11_volum/16 + m3_003_volum/16


#  **********2018年4月，提取方式:历史月份销量比例，考虑时间衰减，月份越近占比越高**********
m4_03 = train_sales_data.loc[(train_sales_data.regYear==2017)&(train_sales_data.regMonth==4) , 'salesVolume'].values / train_sales_data.loc[(train_sales_data.regYear==2017)&(train_sales_data.regMonth==3), 'salesVolume'].values
m4_02 = train_sales_data.loc[(train_sales_data.regYear==2017)&(train_sales_data.regMonth==4) , 'salesVolume'].values / train_sales_data.loc[(train_sales_data.regYear==2017)&(train_sales_data.regMonth==2), 'salesVolume'].values
m4_01 = train_sales_data.loc[(train_sales_data.regYear==2017)&(train_sales_data.regMonth==4) , 'salesVolume'].values / train_sales_data.loc[(train_sales_data.regYear==2017)&(train_sales_data.regMonth==1), 'salesVolume'].values
m4_12 = train_sales_data.loc[(train_sales_data.regYear==2017)&(train_sales_data.regMonth==4) , 'salesVolume'].values / train_sales_data.loc[(train_sales_data.regYear==2016)&(train_sales_data.regMonth==12) , 'salesVolume'].values

m4_004 = train_sales_data.loc[(train_sales_data.regYear==2017)&(train_sales_data.regMonth==4) , 'salesVolume'].values / train_sales_data.loc[(train_sales_data.regYear==2016)&(train_sales_data.regMonth==4) , 'salesVolume'].values

m4_03_volum = test_sales_data.loc[(test_sales_data.regYear==2018)&(test_sales_data.regMonth==3), 'forecastVolum'].values * m4_03
m4_02_volum = test_sales_data.loc[(test_sales_data.regYear==2018)&(test_sales_data.regMonth==2), 'forecastVolum'].values * m4_02
m4_01_volum = test_sales_data.loc[(test_sales_data.regYear==2018)&(test_sales_data.regMonth==1), 'forecastVolum'].values * m4_01
m4_12_volum = train_sales_data.loc[(train_sales_data.regYear==2017)&(train_sales_data.regMonth==12) , 'salesVolume'].values * m4_12
m4_004_volum = train_sales_data.loc[(train_sales_data.regYear==2017)&(train_sales_data.regMonth==4) , 'salesVolume'].values * m4_004

test_sales_data.loc[test_sales_data.regMonth==4, 'forecastVolum'] =  m4_03_volum/2 + m4_02_volum/4 + m4_01_volum/8 + m4_12_volum/16 + m4_004_volum/16

test_sales_data['forecastVolum'] = test_sales_data['forecastVolum'].apply(lambda x: 0 if x < 0 else int(round(x)))
test_sales_data[['id','forecastVolum']].to_csv('./sub_rule.csv',index=False)
