import pandas as pd
import numpy as np
import sys
#规则融合
bb = pd.read_csv('./sub_rule2.csv').sort_values(by='id').reset_index(drop=True)
bb2 = pd.read_csv('../submit/ccf_car_sales_lgb_full.csv').sort_values(by='id').reset_index(drop=True)

pp = pd.DataFrame()
pp['id'] = bb2['id']
pp['forecastVolum'] = bb['forecastVolum']*0.3+bb2['forecastVolum']*0.7
pp['forecastVolum'] = pp['forecastVolum'].round().astype(int)
pp.to_csv('./rule_lgb_sub3.csv', index=False)

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
print(np.mean(pp['forecastVolum']))
print(np.mean(pp[pp['id']<=1342]['forecastVolum']))
print(np.mean(pp[(pp['id']>1342) & (bb['id']<=2684)]['forecastVolum']))
print(np.mean(pp[(pp['id']>2684) & (bb['id']<=4026)]['forecastVolum']))
print(np.mean(pp[(pp['id']>4026)]['forecastVolum']))

'''
430.14166666666665
492.8537878787879
315.7431818181818
443.175
468.794696969697
-----------------------------
442.7477272727273
508.9219696969697
319.16212121212124
464.38257575757575
478.5242424242424  577

427.42556818181816
492.8537878787879
315.7431818181818
443.175
457.93030303030304  579 单模
'''