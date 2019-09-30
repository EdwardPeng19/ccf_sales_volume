import pandas as pd
import numpy as np
import sys
#规则融合
bb = pd.read_csv('./sub_rule2.csv').sort_values(by='id').reset_index(drop=True)
bb2 = pd.read_csv('../submit/ccf_car_sales_lgb_full.csv').sort_values(by='id').reset_index(drop=True)
bb3 = pd.read_csv('./sub_lgb.csv').sort_values(by='id').reset_index(drop=True)

pp = pd.DataFrame()
pp['id'] = bb2['id']
pp['forecastVolum'] = bb['forecastVolum']*0.2 + bb2['forecastVolum']*0.5 + bb3['forecastVolum']*0.3
pp['forecastVolum'] = pp['forecastVolum'].round().astype(int)
pp.to_csv('./rule_lgb_sub5.csv', index=False)

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
'''

