import pandas as pd
from datetime import datetime, timedelta
import re
import os

# 设置当前日期为2025年3月1日
current_date = datetime(2025, 3, 1)


# 读取CSV文件
def convert_youtube_timestamps(input_file, output_file):
    # 读取CSV文件
    df = pd.read_csv(input_file)

    # 确保timestamp列存在
    if 'timestamp' not in df.columns:
        print("错误：CSV文件中没有找到'timestamp'列")
        return

    # 创建一个新列用于存储转换后的年份
    df['year'] = df['timestamp'].apply(lambda x: convert_relative_time_to_year(x, current_date))

    # 保存结果到新的CSV文件
    df.to_csv(output_file, index=False)
    print(f"转换完成，结果已保存到 {output_file}")


def convert_relative_time_to_year(time_str, current_date):
    # 移除"(edited)"标记
    clean_time = re.sub(r'\s*\(edited\)\s*', '', str(time_str))

    # 匹配时间单位和数量
    match = re.match(r'(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+ago', clean_time)

    if not match:
        return None  # 无法解析的时间格式

    amount = int(match.group(1))
    unit = match.group(2)

    # 基于时间单位计算日期
    if unit == 'second':
        converted_date = current_date - timedelta(seconds=amount)
    elif unit == 'minute':
        converted_date = current_date - timedelta(minutes=amount)
    elif unit == 'hour':
        converted_date = current_date - timedelta(hours=amount)
    elif unit == 'day':
        converted_date = current_date - timedelta(days=amount)
    elif unit == 'week':
        converted_date = current_date - timedelta(weeks=amount)
    elif unit == 'month':
        # 大致估算一个月为30天
        converted_date = current_date - timedelta(days=amount * 30)
    elif unit == 'year':
        # 估算一年为365天
        converted_date = current_date - timedelta(days=amount * 365)
    else:
        return None

    # 只返回年份
    return converted_date.year


# 使用示例
if __name__ == "__main__":
    os.chdir(r'C:\Users\mzjj\Desktop\thesis')
    input_file = "./China/merged_comments_with_language.csv"  # 输入文件路径
    output_file = "./China/comments_with_years.csv"  # 输出文件路径
    convert_youtube_timestamps(input_file, output_file)

