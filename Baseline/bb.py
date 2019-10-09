import pandas as pd
import numpy as np
import sys
#规则融合
# bb = pd.read_csv('./sub_rule2.csv').sort_values(by='id').reset_index(drop=True)
# bb2 = pd.read_csv('../submit/ccf_car_sales_lgb_full.csv').sort_values(by='id').reset_index(drop=True)
# bb3 = pd.read_csv('./sub_lgb.csv').sort_values(by='id').reset_index(drop=True)
#
# pp = pd.DataFrame()
# pp['id'] = bb2['id']
# pp['forecastVolum'] = bb['forecastVolum']*0.25 + bb2['forecastVolum']*0.7 + bb3['forecastVolum']*0.05
# pp['forecastVolum'] = pp['forecastVolum'].round().astype(int)
# pp.to_csv('./rule_lgb_sub.csv', index=False)
#
# print('融合平均结果',np.mean(pp['forecastVolum']))
# print(np.mean(pp[pp['id']<=1342]['forecastVolum']))
# print(np.mean(pp[(pp['id']>1342) & (bb['id']<=2684)]['forecastVolum']))
# print(np.mean(pp[(pp['id']>2684) & (bb['id']<=4026)]['forecastVolum']))
# print(np.mean(pp[(pp['id']>4026)]['forecastVolum']))



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


单模计算结果 422.97670454545454
509.05833333333334
310.2659090909091
427.8992424242424
444.68333333333334   0.59774762000


单模计算结果 414.71723484848485
504.84166666666664
304.9318181818182
414.8507575757576
434.244696969697    0.59804147000
-----------------------------  

单模计算结果 416.2066287878788
505.7598484848485
305.1431818181818
417.4219696969697
436.50151515151515
-----------------------------   0.59912419000
 
'''

