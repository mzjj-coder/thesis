# CreatTime : 2025/1/21
# Author : mzjj

'''
实现功能：
    在excel中，对应的名称后面，插入图片

'''

import openpyxl,PIL
from openpyxl.drawing.image import Image
import os
import pandas as pd


# # csv文件路径
# csv_path = './UK/relative_filter.csv'
#
# # csv转Excel
# df = pd.read_csv(csv_path)
# # 添加列名称图片
# df['图片'] = None
# df.to_excel('./UK/image.xlsx', index=False)
#excel文件路径
input_path = './France/without_image.xlsx'
output_path = './France/image.xlsx'
#图片名称为A列
img_name_column = 'L'

#图片写入B列
img_column = 'R'
#读取图片的地址
img_path = './France/thumbnails'

#打开excel文件,忽略头
wb = openpyxl.load_workbook(input_path)
#获取sheet页
ws = wb.active

for i, v in enumerate(ws[img_name_column][1:], start=2):
    #图片路径
    img_file_path = os.path.join(img_path, f"{v.value}.jpg")
    try:
        #获取图片
        img = Image(img_file_path)
        #设置图片的大小
        img.width, img.height = (110, 110)
        # 设置表格的宽20和高85
        ws.column_dimensions[img_column].width = 20
        ws.row_dimensions[i].height = 85
        # 图片插入名称对应单元格
        ws.add_image(img, anchor=img_column + str(i))
    except FileNotFoundError:
        print(f"图片{v.value}.jpg不存在")

#保存
wb.save(output_path)  # 保存
#关闭
wb.close()
print(f'保存完成')


