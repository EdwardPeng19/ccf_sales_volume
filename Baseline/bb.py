import pandas as pd
import numpy as np
import sys
#规则融合
# bb = pd.read_csv('./sub_rule2.csv').sort_values(by='id').reset_index(drop=True)
# bb2 = pd.read_csv('../submit/ccf_car_sales_lgb_full.csv').sort_values(by='id').reset_index(drop=True)
#
# pp = pd.DataFrame()
# pp['id'] = bb2['id']
# pp['forecastVolum'] = bb['forecastVolum']*0.3+bb2['forecastVolum']*0.7
# pp['forecastVolum'] = pp['forecastVolum'].round().astype(int)
# pp.to_csv('./rule_lgb_sub3.csv', index=False)
#
# print(np.mean(pp['forecastVolum']))
# print(np.mean(pp[pp['id']<=1342]['forecastVolum']))
# print(np.mean(pp[(pp['id']>1342) & (bb['id']<=2684)]['forecastVolum']))
# print(np.mean(pp[(pp['id']>2684) & (bb['id']<=4026)]['forecastVolum']))
# print(np.mean(pp[(pp['id']>4026)]['forecastVolum']))

# pp = pd.DataFrame()
# pp['id'] = bb2['id']
# pp['forecastVolum'] = bb2['forecastVolum']
# pp['forecastVolum2'] = bb['forecastVolum']
# pp['forecastVolum'] = pp.apply(lambda x: x['forecastVolum']*0.8+x['forecastVolum2']*0.2 if x['id']>2684 else x['forecastVolum'],axis=1)
# pp['forecastVolum'] = pp['forecastVolum'].round().astype(int)
# pp[['id','forecastVolum']].to_csv('./rule_lgb_sub3.csv', index=False)
#sys.exit(-1)

#结果均值计算
print('-----------------------------')
bb = pd.read_csv('../submit/ccf_car_sales_lgb_full.csv',).sort_values(by='id').reset_index(drop=True)
print(np.mean(bb['forecastVolum']))
print(np.mean(bb[bb['id']<=1342]['forecastVolum']))
print(np.mean(bb[(bb['id']>1342) & (bb['id']<=2684)]['forecastVolum']))
print(np.mean(bb[(bb['id']>2684) & (bb['id']<=4026)]['forecastVolum']))
print(np.mean(bb[(bb['id']>4026)]['forecastVolum']))
print('-----------------------------')


'''
427.42556818181816
492.8537878787879
315.7431818181818
443.175
457.93030303030304  5792 单模

446.6448863636364
518.8583333333333
348.21363636363634
447.35227272727275
472.155303030303  587 融合
----------------------------------

431.92859848484846
503.6628787878788
317.34166666666664
446.7204545454546
459.98939393939395  5793
'''