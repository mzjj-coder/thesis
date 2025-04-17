import pandas as pd
from langdetect import detect
import os
from tqdm import tqdm

# 语言代码到中文名称的映射字典
LANGUAGE_CODE_TO_CHINESE = {
    'af': '南非荷兰语',
    'ar': '阿拉伯语',
    'bg': '保加利亚语',
    'bn': '孟加拉语',
    'ca': '加泰罗尼亚语',
    'cs': '捷克语',
    'cy': '威尔士语',
    'da': '丹麦语',
    'de': '德语',
    'el': '希腊语',
    'en': '英语',
    'es': '西班牙语',
    'et': '爱沙尼亚语',
    'fa': '波斯语',
    'fi': '芬兰语',
    'fr': '法语',
    'gu': '古吉拉特语',
    'he': '希伯来语',
    'hi': '印地语',
    'hr': '克罗地亚语',
    'hu': '匈牙利语',
    'id': '印度尼西亚语',
    'it': '意大利语',
    'ja': '日语',
    'kn': '卡纳达语',
    'ko': '韩语',
    'lt': '立陶宛语',
    'lv': '拉脱维亚语',
    'mk': '马其顿语',
    'ml': '马拉雅拉姆语',
    'mr': '马拉地语',
    'ne': '尼泊尔语',
    'nl': '荷兰语',
    'no': '挪威语',
    'pa': '旁遮普语',
    'pl': '波兰语',
    'pt': '葡萄牙语',
    'ro': '罗马尼亚语',
    'ru': '俄语',
    'sk': '斯洛伐克语',
    'sl': '斯洛文尼亚语',
    'so': '索马里语',
    'sq': '阿尔巴尼亚语',
    'sv': '瑞典语',
    'sw': '斯瓦希里语',
    'ta': '泰米尔语',
    'te': '泰卢固语',
    'th': '泰语',
    'tl': '塔加洛语',
    'tr': '土耳其语',
    'uk': '乌克兰语',
    'ur': '乌尔都语',
    'vi': '越南语',
    'zh': '中文',
    'zh-cn': '中文(简体)',
    'zh-tw': '中文(繁体)',
    'unknown': '未知语言'
}

def detect_language(text):
    """
    检测文本的语言

    参数:
        text (str): 需要检测的文本

    返回:
        str: 检测到的语言代码
    """
    try:
        # 处理空值或非字符串值
        if pd.isna(text) or not isinstance(text, str) or len(text.strip()) == 0:
            return 'unknown'
        if len(text) < 20:  # You can adjust the threshold as needed
            return 'unknown'

        # 检测语言
        lang = detect(text)
        return lang
    except Exception as e:
        # 如果检测失败，返回'unknown'
        print(f"语言检测失败: {e}")
        return 'unknown'

def get_language_name(lang_code):
    """
    将语言代码转换为中文名称

    参数:
        lang_code (str): 语言代码

    返回:
        str: 对应的中文名称
    """
    return LANGUAGE_CODE_TO_CHINESE.get(lang_code, f'{lang_code}(未知语言)')

def process_csv_file(file_path):
    """
    处理CSV文件，添加语言检测列

    参数:
        file_path (str): CSV文件路径
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 检查'text'列是否存在
        if 'text' not in df.columns:
            print(f"错误: 文件 {file_path} 中没有'text'列")
            return

        # 应用语言检测函数
        print("正在检测语言...")
        tqdm.pandas(desc="语言检测")
        df['LanguageCode'] = df['text'].progress_apply(detect_language)

        # 将语言代码转换为中文名称
        df['Language'] = df['LanguageCode'].apply(get_language_name)

        # 保存结果
        output_path = os.path.splitext(file_path)[0] + '_with_language.csv'
        df.to_csv(output_path, index=False)
        print(f"处理完成，结果已保存到: {output_path}")

        # 显示检测到的语言统计
        lang_counts = df['Language'].value_counts()
        print("\n检测到的语言统计:")
        for lang, count in lang_counts.items():
            print(f"{lang}: {count}")

    except Exception as e:
        print(f"处理文件时出错: {e}")

if __name__ == "__main__":
    os.chdir(r'C:\Users\mzjj\Desktop\thesis')
    file_path = "./China/merged_comments.csv"
    process_csv_file(file_path)