import os
import re
import csv
import glob
from tqdm import tqdm

# 语言代码映射
LANGUAGE_MAP = {
    'en': '英语',
    'zh': '中文',
    'zh-CN': '简体中文',
    'zh-TW': '繁体中文',
    'ja': '日语',
    'ko': '韩语',
    'fr': '法语',
    'de': '德语',
    'es': '西班牙语',
    'it': '意大利语',
    'pt': '葡萄牙语',
    'ru': '俄语',
    'ar': '阿拉伯语',
    'hi': '印地语',
    'nl': '荷兰语',
    'sv': '瑞典语',
    'fi': '芬兰语',
    'no': '挪威语',
    'da': '丹麦语',
    'pl': '波兰语',
    'tr': '土耳其语',
    'uk': '乌克兰语',
    'cs': '捷克语',
    'el': '希腊语',
    'he': '希伯来语',
    'hu': '匈牙利语',
    'ro': '罗马尼亚语',
    'th': '泰语',
    'vi': '越南语',
    'id': '印度尼西亚语',
    'ms': '马来语',
    'fa': '波斯语',
    'fil': '菲律宾语',
    'auto': '自动检测'
}

def get_language_name(lang_code):
    """将语言代码转换为完整的语言名称"""
    base_lang = lang_code.split('-')[0]
    return LANGUAGE_MAP.get(base_lang, lang_code)

def extract_language_from_vtt_header(file_path):
    """从VTT文件头部提取语言信息"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for _ in range(5):
                line = file.readline().strip()
                if line.startswith('Language:'):
                    lang_code = line.split(':', 1)[1].strip()
                    return lang_code
    except:
        pass
    return None

def clean_text(text):
    """清理文本，移除标签和时间戳"""
    text = re.sub(r'</?c>', '', text)
    text = re.sub(r'<\d{2}:\d{2}:\d{2}\.\d{3}>', '', text)
    return text.strip()

def extract_text_from_vtt(file_path):
    """提取VTT文件中的纯文本内容，保留时间戳"""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    if 'WEBVTT' in content:
        content = re.sub(r'WEBVTT.*?\n\n', '', content, flags=re.DOTALL)

    content = re.sub(r'Kind:.*?\n', '', content)
    content = re.sub(r'Language:.*?\n', '', content)

    segments = re.split(r'(\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}.*?\n)', content)

    text_segments = []
    last_text = ""

    i = 0
    while i < len(segments):
        if re.match(r'\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}', segments[i]):
            timestamp = segments[i].strip()
            i += 1
            current_segment = segments[i].strip() if i < len(segments) else ""
            if current_segment:
                lines = current_segment.split('\n')
                for line in lines:
                    cleaned_line = clean_text(line)
                    if cleaned_line and cleaned_line != last_text:
                        start, end = timestamp.split(' --> ')
                        end = end.split(' align:start position:0%')[0]
                        text_segments.append((start, end, cleaned_line))
                        last_text = cleaned_line
        i += 1

    return text_segments

def extract_info_from_filename(filename):
    """从文件名提取视频ID和语言信息"""
    base_name = os.path.splitext(filename)[0]
    parts = base_name.split('.')

    if len(parts) >= 2:
        video_id = parts[0]
        lang_part = parts[1].split('-')
        if len(lang_part) >= 1:
            language_code = lang_part[0]
            language_name = get_language_name(language_code)
            is_original = '-orig' in parts[1]
            language = f"{language_name}{'(自动生成)' if is_original else ''}"
        else:
            language = "未知"
    else:
        video_id = base_name
        language = "未知"

    return video_id, language

def process_vtt_files(directory_path, output_csv):
    """处理目录中的所有VTT文件并生成单个CSV文件"""
    vtt_files = glob.glob(os.path.join(directory_path, "*.vtt"))

    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Start', 'End', 'Text', 'Language', 'VideoID'])

        for vtt_file in tqdm(vtt_files, desc="Processing VTT files"):
            filename = os.path.basename(vtt_file)
            video_id, language = extract_info_from_filename(filename)

            try:
                header_lang = extract_language_from_vtt_header(vtt_file)
                if header_lang and header_lang != 'unknown':
                    language_name = get_language_name(header_lang)
                    is_original = '-orig' in filename
                    language = f"{language_name}{'(自动生成)' if is_original else ''}"

                text_segments = extract_text_from_vtt(vtt_file)
                if text_segments:
                    for start, end, text in text_segments:
                        csv_writer.writerow([start, end, text, language, video_id])
                    print(f"已处理: {filename}")
                else:
                    print(f"警告: {filename} 没有提取到文本内容")
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")

def main():
    directory_path = "./China/subtitles"
    output_csv = "./China/merged_subtitles.csv"

    process_vtt_files(directory_path, output_csv)

if __name__ == "__main__":
    main()
