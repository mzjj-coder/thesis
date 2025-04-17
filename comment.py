'''
实现功能：获取评论
利用youtube_comment_downloader
'''


import pandas as pd
from youtube_comment_downloader import *
import os
from datetime import datetime
from tqdm import tqdm
import csv


def write_comment_header(file):
    """写入CSV文件头"""
    headers = ['text', 'author', 'timestamp', 'like_count', 'reply_count', 'channel', 'is_reply', 'video_url',
               'video_id']
    writer = csv.DictWriter(file, fieldnames=headers)
    writer.writeheader()
    return writer


def safe_int_convert(value, default=0):
    """安全地将值转换为整数"""
    if isinstance(value, int):
        return value
    try:
        # 移除任何非数字字符（比如逗号、空格等）
        cleaned = ''.join(c for c in str(value) if c.isdigit())
        return int(cleaned) if cleaned else default
    except (ValueError, TypeError):
        return default


def extract_comment_data(comment, video_url, video_id):
    """安全地提取评论数据"""
    try:
        if isinstance(comment, str):
            return {
                'text': comment,
                'author': '',
                'timestamp': '',
                'like_count': 0,
                'reply_count': 0,
                'channel': '',
                'is_reply': False,
                'video_url': video_url,
                'video_id': video_id
            }

        # 获取点赞数和回复数
        likes = comment.get('votes', '0')
        replies = comment.get('replies', 0)
        reply_count = safe_int_convert(replies)

        return {
            'text': str(comment.get('text', '')),
            'author': str(comment.get('author', '')),
            'timestamp': str(comment.get('time', '')),
            'like_count': safe_int_convert(likes),
            'reply_count': reply_count,
            'channel': str(comment.get('channel', '')),
            'is_reply': reply_count > 0,  # 根据reply_count判断是否有回复
            'video_url': video_url,
            'video_id': video_id
        }
    except Exception as e:
        print(f"处理评论数据时出错: {str(e)}")
        print(f"问题评论数据: {comment}")
        return None


def get_video_comments(url, video_id, output_file):
    """
    使用youtube-comment-downloader获取YouTube视频评论
    实时写入到文件，并显示进度条
    """
    downloader = YoutubeCommentDownloader()
    comments_count = 0
    error_count = 0

    try:
        # 获取评论生成器
        comments_generator = downloader.get_comments_from_url(url, sort_by=SORT_BY_RECENT)

        with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
            writer = write_comment_header(f)

            # 使用tqdm创建进度条
            with tqdm(desc="下载评论", unit="条") as pbar:
                for comment in comments_generator:
                    try:
                        # 提取评论数据
                        comment_data = extract_comment_data(comment, url, video_id)
                        if comment_data:
                            writer.writerow(comment_data)
                            comments_count += 1

                            # 更新进度条
                            pbar.update(1)

                            # 定期刷新文件缓冲区
                            if comments_count % 100 == 0:
                                f.flush()
                                os.fsync(f.fileno())
                    except Exception as e:
                        error_count += 1
                        print(f"处理单条评论时出错: {str(e)}")
                        if error_count <= 5:  # 只显示前5个错误
                            print(f"错误评论数据: {comment}")
                        continue

        if error_count > 0:
            print(f"共有 {error_count} 条评论处理失败")

        return comments_count
    except Exception as e:
        print(f"获取评论时出错 (URL: {url}): {str(e)}")
        return 0


def main():
    # 读取Excel文件
    input_path = os.path.join('./UK', 'without_image.xlsx')
    if not os.path.exists(input_path):
        print("找不到输入文件！")
        return

    df = pd.read_excel(input_path)

    # 确保必要的列存在
    required_columns = ['视频URL', '视频ID']
    if not all(col in df.columns for col in required_columns):
        print("Excel文件缺少必要的列！")
        return

    # 创建输出目录
    output_dir = os.path.join('./UK', 'comments')
    os.makedirs(output_dir, exist_ok=True)

    # 处理每个视频
    total_videos = len(df)
    for index, row in enumerate(df.iterrows(), 1):
        video_url = row[1]['视频URL']
        video_id = row[1]['视频ID']

        print(f"\n处理视频 {index}/{total_videos}: {video_id}")

        # 设置输出文件路径
        output_file = os.path.join(output_dir, f'{video_id}_comments.csv')

        # 获取并写入评论
        comments_count = get_video_comments(video_url, video_id, output_file)

        if comments_count > 0:
            print(f"成功保存 {comments_count} 条评论到 {output_file}")
        else:
            print(f"视频 {video_id} 没有找到评论")


if __name__ == "__main__":
    main()