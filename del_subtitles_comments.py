# CreatTime : 2025/2/27
# Author : mzjj

import os
import pandas as pd
from pathlib import Path


def cleanup_files():
    # 定义基础路径
    base_path = Path('./Japan')

    # 读取Excel文件中的视频ID
    excel_path = base_path / 'without_image.xlsx'
    if not excel_path.exists():
        print(f"错误: 找不到文件 {excel_path}")
        return

    try:
        # 读取Excel文件
        df = pd.read_excel(excel_path)

        # 假设视频ID在一个名为'video_id'的列中，如果不是请修改此行
        if 'video_id' not in df.columns:
            # 尝试找到可能包含视频ID的列
            possible_id_columns = [col for col in df.columns if 'id' in col.lower()]
            if possible_id_columns:
                video_id_column = possible_id_columns[0]
                print(f"未找到'video_id'列，使用'{video_id_column}'列作为视频ID")
            else:
                # 如果没有明显的ID列，使用第一列
                video_id_column = df.columns[0]
                print(f"未找到明显的ID列，使用第一列'{video_id_column}'作为视频ID")
        else:
            video_id_column = 'video_id'

        # 获取所有需要保留的视频ID
        keep_ids = set(df[video_id_column].astype(str))
        print(f"Excel文件中共有 {len(keep_ids)} 个视频ID需要保留")

        # 处理comments文件夹
        comments_path = base_path / 'comments'
        if comments_path.exists():
            process_folder(comments_path, keep_ids, '_comments.csv')
        else:
            print(f"警告: 找不到文件夹 {comments_path}")

        # 处理subtitles文件夹
        subtitles_path = base_path / 'subtitles'
        if subtitles_path.exists():
            # 对于subtitles文件，id可能在文件名前面
            process_folder(subtitles_path, keep_ids, None, id_prefix=True)
        else:
            print(f"警告: 找不到文件夹 {subtitles_path}")

        print("清理完成!")

    except Exception as e:
        print(f"处理文件时出错: {e}")


def process_folder(folder_path, keep_ids, id_suffix=None, id_prefix=False):
    """
    处理文件夹中的文件，删除不在keep_ids中的文件

    参数:
    folder_path: 文件夹路径
    keep_ids: 要保留的视频ID集合
    id_suffix: 视频ID后的后缀 (如 "_comments.csv")
    id_prefix: 如果为True，视频ID在文件名前面而不是后面
    """
    files_to_delete = []

    for file_path in folder_path.iterdir():
        if file_path.is_file():
            file_name = file_path.name

            # 从文件名中提取视频ID
            if id_suffix and file_name.endswith(id_suffix):
                # 例如: "abcdef_comments.csv" -> "abcdef"
                file_id = file_name[:-len(id_suffix)]
            elif id_prefix:
                # 例如: "abcdef.en-orig.vtt" -> "abcdef"
                file_id = file_name.split('.')[0]
            else:
                # 默认情况，假设整个文件名是ID
                file_id = file_path.stem

            # 检查视频ID是否在保留列表中
            if file_id not in keep_ids:
                files_to_delete.append(file_path)

    # 删除不需要保留的文件
    if files_to_delete:
        print(f"\n在 {folder_path} 中发现 {len(files_to_delete)} 个需要删除的文件:")
        for file_path in files_to_delete:
            print(f"将删除: {file_path.name}")

        # 确认删除操作
        confirm = input("\n确认删除以上文件? (y/n): ")
        if confirm.lower() == 'y':
            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                    print(f"已删除: {file_path.name}")
                except Exception as e:
                    print(f"删除 {file_path.name} 时出错: {e}")
        else:
            print("取消删除操作")
    else:
        print(f"在 {folder_path} 中没有需要删除的文件")


if __name__ == "__main__":
    cleanup_files()