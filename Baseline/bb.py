import pandas as pd
import numpy as np
import sys
#规则融合
bb = pd.read_csv('./sub_rule2.csv').sort_values(by='id').reset_index(drop=True)
bb2 = pd.read_csv('../submit/ccf_car_sales_lgb_full.csv').sort_values(by='id').reset_index(drop=True)
bb3 = pd.read_csv('./sub_lgb.csv').sort_values(by='id').reset_index(drop=True)

pp = pd.DataFrame()
pp['id'] = bb2['id']
pp['forecastVolum'] = bb['forecastVolum']*0.2 + bb2['forecastVolum']*0.7 + bb3['forecastVolum']*0.1
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
融合平均结果 433.5299242424242
519.4295454545454
333.7613636363636
431.50454545454545
449.42424242424244   0.60717416

 单模计算结果 417.0153409090909
508.5628787878788
308.630303030303
418.7977272727273
432.07045454545454
-----------------------------  0.60239059
'''

