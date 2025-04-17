# 2025年2月22日
# mzjj

'''
实现功能：
    下载youtube视频的字幕

    youtubeDL 获取的字幕有两种类型：
    1. 手动上传的字幕
    2. 自动生成的字幕
'''
import pandas as pd
from yt_dlp import YoutubeDL
from tqdm import tqdm
import os


def get_original_subtitle_language(video_url):
    """获取带有-orig后缀的字幕语言"""
    info_options = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
    }

    try:
        with YoutubeDL(info_options) as ydl:
            info = ydl.extract_info(video_url, download=False)

            # 检查手动上传的字幕
            if 'subtitles' in info and info['subtitles']:
                orig_langs = [lang for lang in info['subtitles'].keys() if lang.endswith('-orig')]
                if orig_langs:
                    return orig_langs[0]

            # 检查自动生成的字幕
            if 'automatic_captions' in info and info['automatic_captions']:
                orig_langs = [lang for lang in info['automatic_captions'].keys() if lang.endswith('-orig')]
                if orig_langs:
                    return orig_langs[0]

            # 如果没有-orig后缀的，尝试获取'en'字幕
            available_subs = set()
            if 'subtitles' in info and info['subtitles']:
                available_subs.update(info['subtitles'].keys())
            if 'automatic_captions' in info and info['automatic_captions']:
                available_subs.update(info['automatic_captions'].keys())

            if 'en' in available_subs:
                return 'en'

            # 如果还是没有，返回第一个可用的字幕语言
            if available_subs:
                return list(available_subs)[0]

            return None

    except Exception as e:
        print(f"获取字幕信息失败: {str(e)}")
        return None


def download_subtitle(video_url, video_id, base_path='./UK'):
    # 创建保存目录
    subtitle_path = os.path.join(base_path, 'subtitles')
    if not os.path.exists(subtitle_path):
        os.makedirs(subtitle_path)

    # 获取原始字幕语言
    orig_lang = get_original_subtitle_language(video_url)
    if not orig_lang:
        return False

    # 字幕下载配置
    subtitle_options = {
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': [orig_lang],
        'skip_download': True,
        'quiet': True,
        'no_warnings': True,
        'noprogress': True,
        'outtmpl': os.path.join(subtitle_path, f'{video_id}.%(ext)s')
    }

    try:
        # 下载字幕
        with YoutubeDL(subtitle_options) as ydl:
            ydl.download([video_url])
        return True
    except Exception as e:
        return False


def main():
    base_path = './UK'
    # 读取Excel文件
    excel_path = os.path.join(base_path, 'without_image.xlsx')
    df = pd.read_excel(excel_path)

    # 初始化进度条
    pbar = tqdm(total=len(df), desc="处理进度")

    success_count = 0
    fail_count = 0

    for _, row in df.iterrows():
        video_url = row['视频URL']
        video_id = row['视频ID']

        if download_subtitle(video_url, video_id, base_path):
            success_count += 1
        else:
            fail_count += 1

        pbar.update(1)

    pbar.close()

    print(f"\n下载完成！成功: {success_count}, 失败: {fail_count}")


if __name__ == "__main__":
    main()