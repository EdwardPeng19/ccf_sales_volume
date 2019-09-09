import pandas as pd
import numpy as np
import sys
# aa = pd.read_csv('./sub_rule.csv').sort_values(by='id').reset_index()
#
# bb = pd.read_csv('../submit/ccf_car_sales_lgb_20190902211429.csv').sort_values(by='id').reset_index()
# # bb2 = pd.read_csv('../submit/ccf_car_label_lgb_2month.csv').sort_values(by='id').reset_index()
# # print(bb2.head())
# # bb2.columns = ['aa', 'id','num']
# # bb = bb.merge(bb2, how='left', on='id')
# # bb = bb.fillna(0)
# # bb['forecastVolum'] = bb.apply(lambda x:x['forecastVolum'] if x['num']==0 else x['num'], axis=1)
#
# pp = pd.DataFrame()
# pp['id'] = aa['id']
# pp['forecastVolum'] = aa['forecastVolum']*0.3+bb['forecastVolum']*0.7
# pp['forecastVolum'] = pp['forecastVolum'].round().astype(int)
#
# pp.to_csv('./rule_lgb2.csv', index=False)
#
#
# P_SUBMIT = "../submit/"




bb2 = pd.read_csv('../submit/ccf_car_sales_lgb_20190902211429.csv').sort_values(by='id').reset_index(drop=True)
print(np.mean(bb2['forecastVolum']))
print(np.mean(bb2[bb2['id']<=1342]['forecastVolum']))
print(np.mean(bb2[(bb2['id']>1342) & (bb2['id']<=2684)]['forecastVolum']))
print(np.mean(bb2[(bb2['id']>2684) & (bb2['id']<=4026)]['forecastVolum']))
print(np.mean(bb2[(bb2['id']>4026)]['forecastVolum']))
print('-------------------------')

bb = pd.read_csv('./sub_rule.csv',).sort_values(by='id').reset_index(drop=True)
print(np.mean(bb['forecastVolum']))
print(np.mean(bb[bb['id']<=1342]['forecastVolum']))
print(np.mean(bb[(bb['id']>1342) & (bb['id']<=2684)]['forecastVolum']))
print(np.mean(bb[(bb['id']>2684) & (bb['id']<=4026)]['forecastVolum']))
print(np.mean(bb[(bb['id']>4026)]['forecastVolum']))


print('-----------------------------')
bb = pd.read_csv('../submit/ccf_car_sales_lgb_full.csv',).sort_values(by='id').reset_index(drop=True)
print(np.mean(bb['forecastVolum']))
print(np.mean(bb[bb['id']<=1342]['forecastVolum']))
print(np.mean(bb[(bb['id']>1342) & (bb['id']<=2684)]['forecastVolum']))
print(np.mean(bb[(bb['id']>2684) & (bb['id']<=4026)]['forecastVolum']))
print(np.mean(bb[(bb['id']>4026)]['forecastVolum']))


'''
451
'''