# CreatTime : 2025/2/24
# Author : mzjj

'''
利用paraphrase-multilingual-MiniLM-L12-v2多语言模型计算视频标题、描述和标签与关键词的相似度，筛选出与关键词相关的视频
'''
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm

def analyze_video_relevance():
    # 加载本地模型
    model = SentenceTransformer('./paraphrase-multilingual-MiniLM-L12-v2')

    # 读取数据
    df = pd.read_csv('./France/drop_duplicates.csv')

    # 关键词
    keywords = ['France first impression', 'first visit France']

    # 编码关键词
    keyword_embeddings = model.encode(keywords, convert_to_tensor=True)

    # 准备视频文本 (合并标题、描述和标签)
    video_texts = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing videos"):
        text = f"{row['标题']} "  # 标题
        if isinstance(row.get('描述'), str):
            text += f"{row['描述']} "  # 描述
        if isinstance(row.get('标签'), str):
            text += str(row['标签'])  # 标签
        video_texts.append(text)

    # 批量编码视频文本
    video_embeddings = model.encode(video_texts, convert_to_tensor=True, batch_size=32, show_progress_bar=True)

    # 计算相似度
    similarity_scores = util.cos_sim(video_embeddings, keyword_embeddings)

    # 取每个视频与所有关键词的最大相似度
    max_scores = torch.max(similarity_scores, dim=1)[0]

    # 添加相似度分数到DataFrame
    df['相关性'] = max_scores.cpu().numpy()

    # 确定阈值 (这里使用简单的百分位数方法)
    threshold = np.percentile(df['相关性'], 25)  # 保留前75%的视频

    # 添加阈值和是否强相关列
    df['阈值'] = threshold
    df['是否强相关'] = df['相关性'] >= threshold

    # 输出统计信息
    print(f"总视频数: {len(df)}")
    print(f"相关性阈值: {threshold:.3f}")

    # 保存结果到新的CSV文件
    df.to_csv('./France/relative_filter.csv', index=False, encoding='utf-8-sig')

    return df

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("使用 GPU 进行计算")
    else:
        print("使用 CPU 进行计算")

    relevant_videos = analyze_video_relevance()