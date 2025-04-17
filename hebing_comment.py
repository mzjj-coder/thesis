# CreatTime : 2025/2/25
# Author : mzjj

import os
import pandas as pd
import glob

os.chdir(r'C:\Users\mzjj\Desktop\thesis')
# 设置文件夹路径
folder_path = './China/comments'

# 获取该文件夹下所有的CSV文件
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

# 创建一个空的DataFrame用于存储合并后的数据
merged_df = pd.DataFrame()

# 遍历所有CSV文件并合并
for file in csv_files:
    # 获取文件名
    filename = os.path.basename(file)

    # 提取commentID (文件名中_comments前面的部分)
    comment_id = filename.split('_comments')[0]

    # 读取CSV文件
    try:
        df = pd.read_csv(file)

        # 添加commentID列
        df['commentID'] = comment_id

        # 合并到主DataFrame
        if merged_df.empty:
            merged_df = df
        else:
            merged_df = pd.concat([merged_df, df], ignore_index=True)

        print(f"成功处理: {filename}")

    except Exception as e:
        print(f"处理 {filename} 时出错: {str(e)}")

# 如果成功合并了文件，保存结果
if not merged_df.empty:
    output_file = os.path.join('./China', 'merged_comments.csv')
    merged_df.to_csv(output_file, index=False)
    print(f"\n所有CSV文件已合并，结果保存至: {output_file}")
    print(f"总行数: {len(merged_df)}")
else:
    print("没有找到有效的CSV文件或所有文件处理失败")
