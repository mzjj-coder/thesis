import pandas as pd
import re
from textblob import Word
import emoji
import os
import string
import contractions
from tqdm import tqdm
import spacy
from multiprocessing import Pool, cpu_count

# 加载spaCy的英文模型
nlp = spacy.load('en_core_web_sm')

def clean_text(text):
    text = re.sub(r'\d{1,2}:\d{2}', '', text)  # 移除20:26类似这种的时间格式
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)  # 移除URL
    text = re.sub(r'@[\w-]*', '', text)  # 移除@用户名，包括带横线的用户名
    text = re.sub(r'<.*?>', '', text)  # 移除HTML标签
    text = re.sub(r'#\w+', '', text)  # 移除#话题标签
    #删除日期格式
    text = re.sub(r'\d{4}-\d{2}-\d{2}', '', text)  # 移除YYYY-MM-DD格式的日期
    return text

def del_chinese(text):
    if re.search(r'[\u4e00-\u9fa5]', text):
        return ''
    else:
        return text

def clean_emoji(desstr, restr=''):
    try:
        co = re.compile(u'['u'\U0001F300-\U0001F64F' u'\U0001F680-\U0001F6FF'u'\u2600-\u2B55]+')
    except re.error:
        co = re.compile(u'('u'\ud83c[\udf00-\udfff]|'u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'u'[\u2600-\u2B55])+')
    return co.sub(restr, desstr)

def remove_text_emojis(text):
    pattern = r'\b\w+(?:\s*-\s*\w+)*\s*:'
    return re.sub(pattern, '', text)

def remove_fuhao(text):
    return re.sub(r'[^\w\s]', '', text) # 移除特殊符号

def lemmatize_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])

def parallel_lemmatize(texts):
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(lemmatize_text, texts), total=len(texts), desc="Lemmatizing"))
    return results

def normalize_abbreviations(text):
    return contractions.fix(text)

def replace_emoji_with_text(text):
    return emoji.demojize(text)

def remove_punctuation(text):
    return text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))

def preprocess_data(input_file):
    df = pd.read_csv(input_file)
    df = df[df['text'].apply(lambda x: isinstance(x, str))]  # 过滤掉非字符串的值
    df = df[df['Language'] == '英语']  # 只保留英语评论
    df['text_length'] = df['text'].apply(len)
    df_filtered = df[df['text_length'] >= 10]  # 过滤掉长度小于10的评论
    df_filtered['year'] = pd.to_datetime(df_filtered['year'], format='%Y')
    df_new = df_filtered[['year', 'text','Language']].reset_index(drop=True)

    # 基础预处理
    tqdm.pandas(desc="基础预处理")
    df_new['base_clean'] = df_new['text'].progress_apply(lambda x: del_chinese(x))
    df_new['base_clean'] = df_new['base_clean'].progress_apply(lambda x: clean_text(x)) # 移除时间、URL、@用户名和HTML标签
    df_new['base_clean'] = df_new['base_clean'].progress_apply(lambda x: x.lower())  # 大小写还原
    df_new['base_clean'] = df_new['base_clean'].progress_apply(lambda x: normalize_abbreviations(x))  # 归一化缩写

    # 并行处理词形还原
    texts = df_new['base_clean'].tolist()
    df_new['base_clean'] = parallel_lemmatize(texts)

    df_new['base_clean'] = df_new['base_clean'].progress_apply(lambda x: re.sub(r'\s+', ' ', x))  # 去除多个空格

    # 针对BERTopic的预处理
    tqdm.pandas(desc="BERTopic预处理")
    df_new['bertopic_clean'] = df_new['base_clean'].progress_apply(lambda x: remove_text_emojis(x)) # 移除转为文字的表情
    df_new['bertopic_clean'] = df_new['bertopic_clean'].progress_apply(lambda x: clean_emoji(x)) # 去除表情符号
    df_new['bertopic_clean'] = df_new['bertopic_clean'].progress_apply(lambda x: remove_punctuation(x))  # 去除标点符号
    df_new['bertopic_clean'] = df_new['bertopic_clean'].progress_apply(lambda x: remove_fuhao(x)) # 去除特殊符号
    df_new['bertopic_clean'] = df_new['bertopic_clean'].progress_apply(lambda x: re.sub(r'\s+', ' ', x))  # 去除多个空格

    # 针对情感分析的预处理
    tqdm.pandas(desc="情感分析预处理")
    df_new['emotion_clean'] = df_new['base_clean'].progress_apply(lambda x: replace_emoji_with_text(x)) # 替换表情为文字
    df_new['emotion_clean'] = df_new['emotion_clean'].progress_apply(lambda x: remove_punctuation(x))  # 去除标点符号

    df_new['emotion_clean'] = df_new['emotion_clean'].progress_apply(lambda x: re.sub(r'\s+', ' ', x))  # 去除多个空格

    df_new.dropna(subset=['base_clean', 'bertopic_clean', 'emotion_clean'], inplace=True)

    print(len(df_new), "条评论已加载")
    return df_new

if __name__ == '__main__':
    os.chdir(r'C:\Users\mzjj\Desktop\thesis\France')
    input_file = 'comments_with_years.csv'  # 输入文件路径
    df_new = preprocess_data(input_file)  # 预处理数据
    df_new.to_csv('cleaned_comments.csv', index=False, encoding='utf-8-sig')  # 保存预处理后的数据
    print("预处理完成，已保存到 cleaned_comments.csv")
