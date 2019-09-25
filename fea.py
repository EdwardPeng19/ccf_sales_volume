# 销量

sales_thQuarter = []
for sea in [1, 2, 3, 4]:
    sales_thQuarter.extend([f'sales_{sea}thQuarter_var', f'sales_{sea}thQuarter_mean', f'sales_{sea}thQuarter_min',
                            f'sales_{sea}thQuarter_max'])
sales_thYear = []
for hy in [1, 2]:
    sales_thYear.extend([f'sales_{hy}thHalfyear_var', f'sales_{hy}thHalfyear_mean', f'sales_{hy}thHalfyear_min',
                         f'sales_{hy}thHalfyear_max'])
sales_Year = ['sales_Year_var', 'sales_Year_mean', 'sales_Year_min', 'sales_Year_max']

# 销量 车型

sales_model_thQuarter = []
for sea in [1, 2, 3, 4]:
    sales_model_thQuarter.extend([f'sales_model_{sea}thQuarter_var', f'sales_model_{sea}thQuarter_mean', f'sales_model_{sea}thQuarter_min',f'sales_model_{sea}thQuarter_max'])
sales_model_thYear = []
for hy in [1, 2]:
    sales_model_thYear.extend(
        [f'sales_model_{hy}thHalfyear_var', f'sales_model_{hy}thHalfyear_mean', f'sales_model_{hy}thHalfyear_min',
         f'sales_model_{hy}thHalfyear_max'])
sales_model_Year = ['sales_model_Year_var', 'sales_model_Year_mean', 'sales_model_Year_min', 'sales_model_Year_max']

# 销量 省份

sales_pro_thQuarter = []
for sea in [1, 2, 3, 4]:
    sales_pro_thQuarter.extend(
        [f'sales_pro_{sea}thQuarter_var', f'sales_pro_{sea}thQuarter_mean', f'sales_pro_{sea}thQuarter_min',
         f'sales_pro_{sea}thQuarter_max'])
sales_pro_thYear = []
for hy in [1, 2]:
    sales_pro_thYear.extend(
        [f'sales_pro_{hy}thHalfyear_var', f'sales_pro_{hy}thHalfyear_mean', f'sales_pro_{hy}thHalfyear_min',
         f'sales_pro_{hy}thHalfyear_max'])
sales_pro_Year = ['sales_pro_Year_var', 'sales_pro_Year_mean', 'sales_pro_Year_min', 'sales_pro_Year_max']

# 销量 车身

sales_body_thQuarter = []
for sea in [1, 2, 3, 4]:
    sales_body_thQuarter.extend(
        [f'sales_body_{sea}thQuarter_var', f'sales_body_{sea}thQuarter_mean', f'sales_body_{sea}thQuarter_min',
         f'sales_body_{sea}thQuarter_max'])
sales_body_thYear = []
for hy in [1, 2]:
    sales_body_thYear.extend(
        [f'sales_body_{hy}thHalfyear_var', f'sales_body_{hy}thHalfyear_mean', f'sales_body_{hy}thHalfyear_min',
         f'sales_body_{hy}thHalfyear_max'])
sales_body_Year = ['sales_body_Year_var', 'sales_body_Year_mean', 'sales_body_Year_min', 'sales_body_Year_max']

# 搜索量

popularity_thQuarter = []
for sea in [1, 2, 3, 4]:
    popularity_thQuarter.extend(
        [f'popularity_{sea}thQuarter_var', f'popularity_{sea}thQuarter_mean', f'popularity_{sea}thQuarter_min',
         f'popularity_{sea}thQuarter_max'])
popularity_thYear = []
for hy in [1, 2]:
    popularity_thYear.extend(
        [f'popularity_{hy}thHalfyear_var', f'popularity_{hy}thHalfyear_mean', f'popularity_{hy}thHalfyear_min',
         f'popularity_{hy}thHalfyear_max'])
popularity_Year = ['popularity_Year_var', 'popularity_Year_mean', 'popularity_Year_min', 'popularity_Year_max']

# 搜索量 车型

popularity_model_thQuarter = []
for sea in [1, 2, 3, 4]:
    popularity_model_thQuarter.extend([f'popularity_model_{sea}thQuarter_var', f'popularity_model_{sea}thQuarter_mean',
                                       f'popularity_model_{sea}thQuarter_min', f'popularity_model_{sea}thQuarter_max'])
popularity_model_thYear = []
for hy in [1, 2]:
    popularity_model_thYear.extend([f'popularity_model_{hy}thHalfyear_var', f'popularity_model_{hy}thHalfyear_mean',
                                    f'popularity_model_{hy}thHalfyear_min', f'popularity_model_{hy}thHalfyear_max'])
popularity_model_Year = ['popularity_model_Year_var', 'popularity_model_Year_mean', 'popularity_model_Year_min',
                         'popularity_model_Year_max']

# 搜索量 省份

popularity_pro_thQuarter = []
for sea in [1, 2, 3, 4]:
    popularity_pro_thQuarter.extend([f'popularity_pro_{sea}thQuarter_var', f'popularity_pro_{sea}thQuarter_mean',
                                     f'popularity_pro_{sea}thQuarter_min', f'popularity_pro_{sea}thQuarter_max'])
popularity_pro_thYear = []
for hy in [1, 2]:
    popularity_pro_thYear.extend([f'popularity_pro_{hy}thHalfyear_var', f'popularity_pro_{hy}thHalfyear_mean',
                                  f'popularity_pro_{hy}thHalfyear_min', f'popularity_pro_{hy}thHalfyear_max'])
popularity_pro_Year = ['popularity_pro_Year_var', 'popularity_pro_Year_mean', 'popularity_pro_Year_min',
                       'popularity_pro_Year_max']

# 搜索量 车身

popularity_body_thQuarter = []
for sea in [1, 2, 3, 4]:
    popularity_body_thQuarter.extend([f'popularity_body_{sea}thQuarter_var', f'popularity_body_{sea}thQuarter_mean',
                                      f'popularity_body_{sea}thQuarter_min', f'popularity_body_{sea}thQuarter_max'])
popularity_body_thYear = []
for hy in [1, 2]:
    popularity_body_thYear.extend([f'popularity_body_{hy}thHalfyear_var', f'popularity_body_{hy}thHalfyear_mean',
                                   f'popularity_body_{hy}thHalfyear_min', f'popularity_body_{hy}thHalfyear_max'])
popularity_body_Year = ['popularity_body_Year_var', 'popularity_body_Year_mean', 'popularity_body_Year_min',
                        'popularity_body_Year_max']

# 评论

comment_thQuarter = []
for sea in [1, 2, 3, 4]:
    comment_thQuarter.extend(
        [f'comment_{sea}thQuarter_var', f'comment_{sea}thQuarter_mean', f'comment_{sea}thQuarter_min',
         f'comment_{sea}thQuarter_max'])
comment_thYear = []
for hy in [1, 2]:
    comment_thYear.extend([f'comment_{hy}thHalfyear_var', f'comment_{hy}thHalfyear_mean', f'comment_{hy}thHalfyear_min',
                           f'comment_{hy}thHalfyear_max'])
comment_Year = ['comment_Year_var', 'comment_Year_mean', 'comment_Year_min', 'comment_Year_max']

# 回复

reply_thQuarter = []
for sea in [1, 2, 3, 4]:
    reply_thQuarter.extend([f'reply_{sea}thQuarter_var', f'reply_{sea}thQuarter_mean', f'reply_{sea}thQuarter_min',
                            f'reply_{sea}thQuarter_max'])
reply_thYear = []
for hy in [1, 2]:
    reply_thYear.extend([f'reply_{hy}thHalfyear_var', f'reply_{hy}thHalfyear_mean', f'reply_{hy}thHalfyear_min',
                         f'reply_{hy}thHalfyear_max'])
reply_Year = ['reply_Year_var', 'reply_Year_mean', 'reply_Year_min', 'reply_Year_max']

sales_features = sales_thMonth + sales_thQuarter + sales_thYear + sales_Year
sales_model_features = sales_model_thMonth + sales_model_thQuarter + sales_model_thYear + sales_model_Year
sales_pro_features = sales_pro_thMonth + sales_pro_thQuarter + sales_pro_thYear + sales_pro_Year
sales_body_features = sales_body_thMonth + sales_body_thQuarter + sales_body_thYear + sales_body_Year
popularity_features = popularity_thMonth + popularity_thQuarter + popularity_thYear + popularity_Year
popularity_model_features = popularity_model_thMonth + popularity_model_thQuarter + popularity_model_thYear + popularity_model_Year
popularity_pro_features = popularity_pro_thMonth + popularity_pro_thQuarter + popularity_pro_thYear + popularity_pro_Year
popularity_body_features = popularity_body_thMonth + popularity_body_thQuarter + popularity_body_thYear + popularity_body_Year

comment_features = comment_thMonth + comment_thQuarter + comment_thYear + comment_Year
reply_features = reply_thMonth + reply_thQuarter + reply_thYear + reply_Year
diff_features = []
time_features = []
for i in range(2, 12):
    diff_features.append(f'sales_diff_{i}_{i - 1}')
    # diff_features.append(f'popularity_diff_{i}_{i - 1}')
    diff_features.append(f'comment_diff_{i}_{i - 1}')
    diff_features.append(f'reply_diff_{i}_{i - 1}')
    diff_features.append(f'sales_model_diff_{i}_{i - 1}')
    # diff_features.append(f'popularity_model_{i}_{i - 1}')
    diff_features.append(f'sales_pro_diff_{i}_{i - 1}')
    # diff_features.append(f'popularity_pro_diff_{i}_{i - 1}')
    diff_features.append(f'sales_body_diff_{i}_{i - 1}')
    # diff_features.append(f'popularity_body_diff_{i}_{i - 1}')

    time_features.append(f'sales_time_{i}_{i - 1}')
    # time_features.append(f'popularity_time_{i}_{i - 1}')
    time_features.append(f'comment_time_{i}_{i - 1}')
    time_features.append(f'reply_time_{i}_{i - 1}')
    time_features.append(f'sales_model_time_{i}_{i - 1}')
    # time_features.append(f'popularity_model_{i}_{i - 1}')
    time_features.append(f'sales_pro_time_{i}_{i - 1}')
    # time_features.append(f'popularity_pro_time_{i}_{i - 1}')
    time_features.append(f'sales_body_time_{i}_{i - 1}')
    # time_features.append(f'popularity_body_time_{i}_{i - 1}')