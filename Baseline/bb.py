import pandas as pd
import numpy as np
import sys
#规则融合
bb = pd.read_csv('./sub_rule2.csv').sort_values(by='id').reset_index(drop=True)
bb2 = pd.read_csv('../submit/ccf_car_sales_lgb_full.csv').sort_values(by='id').reset_index(drop=True)
bb3 = pd.read_csv('./sub_lgb.csv').sort_values(by='id').reset_index(drop=True)

pp = pd.DataFrame()
pp['id'] = bb2['id']
pp['forecastVolum'] = bb['forecastVolum']*0.25 + bb2['forecastVolum']*0.7 + bb3['forecastVolum']*0.05
pp['forecastVolum'] = pp['forecastVolum'].round().astype(int)
pp.to_csv('./rule_lgb_sub.csv', index=False)

print('融合平均结果',np.mean(pp['forecastVolum']))
print(np.mean(pp[pp['id']<=1342]['forecastVolum']))
print(np.mean(pp[(pp['id']>1342) & (bb['id']<=2684)]['forecastVolum']))
print(np.mean(pp[(pp['id']>2684) & (bb['id']<=4026)]['forecastVolum']))
print(np.mean(pp[(pp['id']>4026)]['forecastVolum']))



#结果均值计算
bb = pd.read_csv('../submit/ccf_car_sales_lgb_full.csv',).sort_values(by='id').reset_index(drop=True)
print('单模计算结果',np.mean(bb['forecastVolum']))
print(np.mean(bb[bb['id']<=1342]['forecastVolum']))
print(np.mean(bb[(bb['id']>1342) & (bb['id']<=2684)]['forecastVolum']))
print(np.mean(bb[(bb['id']>2684) & (bb['id']<=4026)]['forecastVolum']))
print(np.mean(bb[(bb['id']>4026)]['forecastVolum']))
print('-----------------------------')


'''
融合平均结果 435.8863636363636
503.6901515151515
335.7613636363636
447.6128787878788
456.4810606060606   598
单模计算结果 415.1806818181818
490.37348484848485
304.3689393939394
430.73939393939395
435.2409090909091   593
-----------------------------

单模计算结果 422.63863636363635
520.1598484848485
301.2409090909091
432.5681818181818
436.58560606060604
-----------------------------  0.59739631000

feature process use time 18.089632511138916
特征数量为44
LabelEncoder Successfully!
train_mae:0.13099747474747475,train_mse:0.13339646464646465
month 25: 520.1598484848485
feature process use time 20.59395670890808
特征数量为19
LabelEncoder Successfully!
train_mae:0.12727272727272726,train_mse:0.1294871794871795
month 26: 301.2409090909091
feature process use time 20.76348567008972
特征数量为65
LabelEncoder Successfully!
train_mae:0.12516233766233767,train_mse:0.1268939393939394
month 27: 432.5681818181818
feature process use time 25.017690658569336
特征数量为201
LabelEncoder Successfully!
train_mae:0.11055555555555556,train_mse:0.11217171717171717
month 28: 436.58560606060604
feature process use time 22.580257415771484

单模计算结果 422.97670454545454
509.05833333333334
310.2659090909091
427.8992424242424
444.68333333333334   0.59774762000

feature process use time 19.210635900497437
特征数量为44
LabelEncoder Successfully!
train_mae:0.13150252525252526,train_mse:0.13390151515151516
month 25: 509.05833333333334
feature process use time 20.84626054763794
特征数量为19
LabelEncoder Successfully!
train_mae:0.12703962703962704,train_mse:0.12937062937062938
month 26: 310.2659090909091
feature process use time 21.338971853256226
特征数量为65
LabelEncoder Successfully!
train_mae:0.12505411255411256,train_mse:0.12667748917748917
month 27: 427.8992424242424
feature process use time 21.35988759994507
特征数量为201
LabelEncoder Successfully!
train_mae:0.11282828282828283,train_mse:0.11454545454545455
month 28: 444.68333333333334
feature process use time 21.490512132644653


单模计算结果 414.71723484848485
504.84166666666664
304.9318181818182
414.8507575757576
434.244696969697    0.59804147000
-----------------------------  
'''

