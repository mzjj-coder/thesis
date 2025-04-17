# CreatTime : 2025/2/26
# Author : mzjj


import pandas as pd
df = pd.read_csv('./France/all_data.csv')

# 重复值检测
df_1 = df.copy()
df_1.drop_duplicates(inplace=True)

# 保存结果
df_1.to_csv('./France/drop_duplicates.csv', index=None, encoding='utf-8-sig')
print('重复值已删除{}条'.format(len(df) - len(df_1)))