# CreatTime : 2025/3/23
# Author : mzjj

import pandas as pd
from datetime import timedelta

# 读取 CSV 文件
file_path = "./France/merged_subtitles.csv"  # 替换为你的文件路径
subtitle = pd.read_csv(file_path)

# 确保时间字段为 timedelta 格式
subtitle["Start"] = pd.to_timedelta(subtitle["Start"])
subtitle["End"] = pd.to_timedelta(subtitle["End"])

# 定义时间窗口大小
window_size = timedelta(seconds=20)


def segment_by_time(group):
    """对同一个 VideoID 的字幕进行时间窗口划分"""
    grouped = []
    temp_text = []
    start_time = group.iloc[0]["Start"]
    language = group.iloc[0]["Language"]
    video_id = group.iloc[0]["VideoID"]

    for _, row in group.iterrows():
        if row["Start"] - start_time > window_size:
            grouped.append({"Text": " ".join(temp_text), "Language": language, "VideoID": video_id})
            temp_text = []
            start_time = row["Start"]
        temp_text.append(row["Text"])

    if temp_text:
        grouped.append({"Text": " ".join(temp_text), "Language": language, "VideoID": video_id})

    return pd.DataFrame(grouped)


# 按 VideoID 分组并应用切分逻辑
segmented_subtitles = subtitle.groupby("VideoID", group_keys=False).apply(segment_by_time)

# 重置索引
segmented_subtitles.reset_index(drop=True, inplace=True)

# 保存结果
segmented_subtitles.to_csv("./France/segmented_subtitles.csv", index=False, encoding='utf-8-sig')
print("切分完成，结果已保存。")

