# 2025年1月21日
# mzjj
# yt_dlp爬取视频元数据（标题、上传者、上传日期、观看次数、点赞数、评论数、时长、频道URL、标签、视频URL、视频ID、缩略图、描述）
# 不含字幕、视频、评论


from yt_dlp import YoutubeDL
from typing import List, Dict, Any, Optional
import time
import csv
import os
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

class YouTubeMetadataFetcher:
    def __init__(self, output_dir="UK",start_date: Optional[str] = None,
                      end_date: Optional[str] = None,):
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "thumbnails")
        self.csv_path = os.path.join(output_dir, f"{start_date}-{end_date}.csv")

        # 创建必要的目录
        for dir_path in [self.output_dir, self.images_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        # 设置代理
        os.environ['HTTP_PROXY'] = 'http://127.0.0.1:33210'
        os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:33210'

        self.ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'forcejson': True,
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'no_check_certificate': True,
        }

        # 初始化 CSV 文件并写入表头
        with open(self.csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            fieldnames = [
                '序号', '标题', '上传者', '上传日期', '观看次数', '点赞数','评论数',
                '时长(秒)', '频道URL', '标签', '视频URL', '视频ID', '缩略图路径',
                '描述',
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    def download_thumbnail(self, url: str, video_id: str) -> str:
        """下载并保存缩略图"""
        if not url:
            return ""
        try:
            response = requests.get(url)
            if response.status_code == 200:
                img_path = os.path.join(self.images_dir, f"{video_id}.jpg")
                Image.open(BytesIO(response.content)).save(img_path)
                return img_path
        except Exception as e:
            print(f"下载缩略图时出错: {str(e)}")
        return ""

    def save_to_csv(self, video_data: Dict) -> None:
        """保存单条数据到CSV文件"""
        try:
            with open(self.csv_path, 'a', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=video_data.keys())
                writer.writerow(video_data)
        except Exception as e:
            print(f"保存CSV时出错: {str(e)}")

    def search_videos(self,
                      keywords: List[str],
                      max_results: int = 5,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      )-> List[Dict[Any, Any]]:

        # 转换日期格式
        start_date_formatted = format_date_for_youtube(start_date)
        end_date_formatted = format_date_for_youtube(end_date)

        search_query = ' '.join(keywords)
        search_query = f"{search_query} before:{end_date_formatted} after:{start_date_formatted}"



        try:
            search_url = f"ytsearch{max_results}:{search_query}"

            ydl_opts = self.ydl_opts.copy()
            if start_date:
                ydl_opts['dateafter'] = start_date
            if end_date:
                ydl_opts['datebefore'] = end_date

            with YoutubeDL(ydl_opts) as ydl:
                results = ydl.extract_info(search_url, download=False)

            if 'entries' not in results:
                return []

            for index, video in enumerate(tqdm(results['entries'], desc="Processing videos"), 1):
                if video:
                    try:
                        # 下载缩略图
                        thumbnail_path = self.download_thumbnail(
                            video.get('thumbnail', ''),
                            video.get('id', '')
                        )

                        formatted_info = {
                            '序号': index,
                            '标题': video.get('title', 'Unknown'),
                            '上传者': video.get('uploader', 'Unknown'),
                            '上传日期': video.get('upload_date', 'Unknown'),
                            '观看次数': video.get('view_count', 'Unknown'),
                            '点赞数': video.get('like_count', 'Unknown'),
                            '评论数': video.get('comment_count', 'Unknown'),
                            '时长(秒)': video.get('duration', 'Unknown'),
                            '频道URL': video.get('channel_url', 'Unknown'),
                            '标签': ', '.join(video.get('tags', [])) if video.get('tags') else '',
                            '视频URL': f"https://www.youtube.com/watch?v={video.get('id')}",
                            '视频ID': video.get('id', ''),
                            '缩略图路径': thumbnail_path,
                            '描述': video.get('description', 'No description'),
                        }

                        # 保存单条数据到CSV
                        self.save_to_csv(formatted_info)

                        # 添加延迟
                        time.sleep(0.5)

                    except Exception as e:
                        print(f"处理视频信息时出错: {str(e)}")
                        continue

            return []

        except Exception as e:
            print(f"搜索过程中出错: {str(e)}")
            return []

def format_date_for_youtube(date_str):
    """将 YYYYMMDD 格式转换为 YYYY-MM-DD 格式"""
    return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"

# 使用示例
if __name__ == "__main__":
    # 设置时间范围
    end_date = '20250101'
    start_date = '20240101'

    # 创建爬取器实例
    fetcher = YouTubeMetadataFetcher(output_dir="France", start_date=start_date, end_date=end_date)

    # 设置搜索关键词
    keywords = ["first visit France", "France first impression"]



    # 获取视频数据
    fetcher.search_videos(keywords, max_results=50, start_date=start_date, end_date=end_date)