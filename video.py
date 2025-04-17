# CreatTime : 2025/2/22
# Author : mzjj

import pandas as pd
from yt_dlp import YoutubeDL
from tqdm import tqdm
import os


def download_video(video_url, video_id, base_path='./youtube_data'):
    # 创建保存目录
    video_path = os.path.join(base_path, 'videos')

    if not os.path.exists(video_path):
        os.makedirs(video_path)


    # 视频下载配置
    video_options = {
        'format': 'best',
        'outtmpl': os.path.join(video_path, f'{video_id}.%(ext)s'),
        'noplaylist': True
    }

    try:

        # 下载视频
        with YoutubeDL(video_options) as ydl:
            ydl.download([video_url])
        return True
    except Exception as e:
        print(f"下载失败 {video_id}: {str(e)}")
        return False


def main():
    base_path = './youtube_data'
    # 读取Excel文件
    excel_path = os.path.join(base_path, 'your_excel_file.xlsx')  # 替换为你的xlsx文件名
    df = pd.read_excel(excel_path)

    # 初始化进度条
    pbar = tqdm(total=len(df), desc="处理进度")

    success_count = 0
    fail_count = 0

    for _, row in df.iterrows():
        video_url = row['视频URL']
        video_id = row['视频ID']

        if download_video(video_url, video_id, base_path):
            success_count += 1
        else:
            fail_count += 1

        pbar.update(1)

    pbar.close()

    print(f"\n下载完成！成功: {success_count}, 失败: {fail_count}")


if __name__ == "__main__":
    main()