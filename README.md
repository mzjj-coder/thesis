# 文档说明

## 1.YouTube crawler
yt_dlp爬取视频元数据（标题、上传者、上传日期、观看次数、点赞数、时长、频道URL、标签、视频URL、视频ID、缩略图、描述）

不含字幕、视频、评论

目前必须限制爬取数量，而且会强制返回设定数量视频，不足则降低相关性，max_results设定不能太大，否则响应速度慢
获得文件夹youtube_data，包含csv文件和缩略图

合并.py 合并后获得all_data.csv
drop_duplicates.py 去重后获得drop_duplicates.csv

## 2.relation_filter
利用paraphrase-multilingual-MiniLM-L12-v2多语言模型计算视频标题、描述和标签与关键词的相似度，筛选出与关键词相关的视频
计算后获得relation_filter.csv

## 3.add picture
添加缩略图到Excel中
便于后续筛选视频
image.xlsx
人工筛选:
- 极少评论（<5)、点赞的(但也有例外，比如很多点赞，但关闭评论区了，仍然保留其作为博主认知的subtitles)
- 相关性
- 时长（但也有例外，比如视频虽然不到1min，但是有很多字幕，有很多点赞评论的，予以保留）
- 带有travel guide的视情况予以删除(但也有例外，比如外国人拍摄的，予以保留)
- 根据评论数量倒序排列，看尤其关注很多评论的视频是否相关

筛选后重新add picture
删除图片without_image .xlsx

## 4.重新保存缩略图
在线Excel图片提取工具，可以一键提取Excel中包含的所有图片
https://uutool.cn/excel-img/ 封面-image 文件夹
手动提取Excel图片方法：将Excel文件后缀.xlsx改成.zip，然后使用压缩软件打开，解压出xl/media/路径下所有图片即可。

## 5.subtitle
字幕爬取
hebing_subtitle.py 合并字幕 提取时间戳，输出merged_subtitles.csv，（字段：Start,End,Text,Language,VideoID）便于后续划分文档
split_subtitle.py 划分文档，便于主题建模，输出segmented_subtitles.csv（字段：Text,Language,VideoID）
字幕分为三种 1.自动生成的 YouTube识别视频的语言自动生成 2. 用户上传的 3.用户视频嵌入图像形式的，无法提取


## 6.comment
评论爬取
只获取了一级评论
hebing_comment.py 合并评论

## 7.del_subtitles_comments
如果后续重复再删除image.xlsx了
就需要删除多余的字幕和评论

## 8. description
数据描述性分析

## 9. detect_comment_language
检测评论语言，输出merged_comments_with_language.csv

## 10. comment_process
处理时间，输出comments_with_years.csv

## 11. text_process




