while 1:
    print('请选择模式：\n1.输入图片生成表情包。\n2.根据人物关键词生成表情包。')
    mode_all=input()
    if mode_all =='1':
    
        break;
    if mode_all =='2':
        break;
    
    print('请先输入1或2选择模式！\n')
    
if mode_all=='1':
    print('请以‘img.jpg’的文件名把图片拖入本文件夹')
    word='img.jpg'
if mode_all=='2':
    print('请输入人物关键词（比如林黛玉，科比，C罗等）')
    word=input()

import torch  #调用文生图模型
import random
import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
keyword=word
if mode_all=='2':
    task = Tasks.text_to_image_synthesis
    model_id = 'damo/multi-modal_chinese_stable_diffusion_v1.0'
    # 基础调用
    pipe = pipeline(task=task, model=model_id)
    output = pipe({'text': word})
    cv2.imwrite(f'{word}_1.jpg', output['output_imgs'][0])
# 输出为opencv numpy格式，转为PIL.Image
# from PIL import Image
# img = output['output_imgs'][0]
# img = Image.fromarray(img[:,:,::-1])
# img.save('result.png')

# 更多参数
if mode_all=='2':
    pipe = pipeline(task=task, model=model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    output = pipe({'text': word, 'num_inference_steps':38, 'guidance_scale': 7.5, 'negative_prompt':'模糊的'})    #这里可以调参，这里进行了微调，步长由50变成了38
    cv2.imwrite(f'{word}_2.jpg', output['output_imgs'][0])

    # 采用DPMSolver
    from diffusers.schedulers import DPMSolverMultistepScheduler
    pipe = pipeline(task=task, model=model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    pipe.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.pipeline.scheduler.config)
    output = pipe({'text': word, 'num_inference_steps': 25})
    cv2.imwrite(f'{word}_2.jpg', output['output_imgs'][0])


wordlst=[]
result_lst=[]
emotionlst=[]
for i in range(2):
    newname=word+'_'+str(i+1)+'.jpg'
    wordlst.append(newname)
print(wordlst)

from modelscope.pipelines import pipeline   #表情识别
from modelscope.utils.constant import  Tasks
import numpy as np
if mode_all=='2':
    try:
        for i in range(2):
            fer = pipeline(Tasks.facial_expression_recognition, 'damo/cv_vgg19_facial-expression-recognition_fer')
            img_path = wordlst[i]
            ret = fer(img_path)
            label_idx = np.array(ret['scores']).argmax()
            label = ret['labels'][label_idx]
            emotionlst.append(label)
            print(f'facial expression : {label}.')
        print(emotionlst)
    except:
        ii1=random.randint(0,6)
        ii2=random.randint(0,6)
        lllst=['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
        emotionlst.append(lllst[ii1])
        emotionlst.append(lllst[ii2])
        print(emotionlst)
if mode_all=='1':
    try:
        fer = pipeline(Tasks.facial_expression_recognition, 'damo/cv_vgg19_facial-expression-recognition_fer')
        img_path ='img.jpg'
        ret = fer(img_path)
        label_idx = np.array(ret['scores']).argmax()
        label = ret['labels'][label_idx]
        emotionlst.append(label)
        print(f'facial expression : {label}.')
        print(emotionlst)
    except:
        ii1=random.randint(0,6)
        
        lllst=['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
        emotionlst.append(lllst[ii1])
        
        print(emotionlst)
#coding=gbk
import csv
class 列:
    def __init__(self,name):
        self.标题=None
        self.name=name
        self.第几列=None
        self.lst=[]
    def pr(self):
        print(self.name)
        print(self.lst)
列集合=[]
# 方法一：使用csv模块，逐行读取
with open('文本库.csv', 'r', encoding='gbk',newline='') as file:
    reader = csv.reader(file)
    t=0
    for row in reader:
        if t==0:
           for i in range(len(row)):
               列集合.append(列(row[i]))
        if t>0:
            for i in range(len(row)):
                列集合[i].lst.append(row[i])
        t+=1
列集合[0].pr()
列集合[1].pr()

列集合[0].pr()
列集合[1].pr()
列集合[2].pr()
列集合[3].pr()
列集合[4].pr()
列集合[5].pr()
列集合[6].pr()

if mode_all=='1':
    #处理为卡通图片
    import cv2
    from modelscope.outputs import OutputKeys
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    img_cartoon = pipeline(Tasks.image_portrait_stylization, 
                           model='damo/cv_unet_person-image-cartoon_compound-models')
    # 图像本地路径
    #img_path = 'input.png'
    # 图像url链接
    img_path = 'img.jpg'
    result = img_cartoon(img_path)
    cv2.imwrite('result.png', result[OutputKeys.OUTPUT_IMG])
    print('finished!')

import random
result_lst=[]
if mode_all=='1':
    i=random.randint(0,8)
    if label=='Angry':
        output_text=列集合[0].lst[i]
    if label=='Disgust':
        output_text=列集合[1].lst[i]
    if label=='Fear':
        output_text=列集合[2].lst[i]
    if label=='Happy':
        output_text=列集合[3].lst[i]
    if label=='Sad':
        output_text=列集合[4].lst[i]
    if label=='Surprise':
        output_text=列集合[5].lst[i]
    if label=='Neutral':
        output_text=列集合[6].lst[i]
    result_lst.append(output_text)
    print(result_lst)
if mode_all=='2':
    for _ in range(2):
        i=random.randint(0,9)
        label=emotionlst[_]
        print(label)
        if label=='Angry':
            output_text=列集合[0].lst[i]
        if label=='Disgust':
            output_text=列集合[1].lst[i]
        if label=='Fear':
            output_text=列集合[2].lst[i]
        if label=='Happy':
            output_text=列集合[3].lst[i]
        if label=='Sad':
            output_text=列集合[4].lst[i]
        if label=='Surprise':
            output_text=列集合[5].lst[i]
        if label=='Neutral':
            output_text=列集合[6].lst[i]
        result_lst.append(output_text)
    print(result_lst)

#图片添加文字
from PIL import Image, ImageDraw, ImageFont
i=0
if mode_all=='2':
    for _ in wordlst:
    # 打开图片
        image = Image.open(_)

        if image.mode == 'P' or image.mode == 'RGBA':
            image = image.convert('RGB')

        # 创建一个绘制对象
        draw = ImageDraw.Draw(image)

        # 设置要添加的文字内容、字体、字号和颜色
        text =result_lst[i]
        font = ImageFont.truetype("simhei.ttf", 125)
        color = (255, 0, 0)  # 白色

        # 在指定位置添加文字
        draw.text((30, image.size[1]//2), text, font=font, fill=color)

        # 保存修改后的图片
        image.save(f'resutl_{i+1}.jpg')
        image.show()
        i+=1
if mode_all=='1':
        image = Image.open('result.png')

        if image.mode == 'P' or image.mode == 'RGBA':
            image = image.convert('RGB')

        # 创建一个绘制对象
        draw = ImageDraw.Draw(image)

        # 设置要添加的文字内容、字体、字号和颜色
        text =result_lst[i]
        font = ImageFont.truetype("simhei.ttf", 75)
        color = (255, 0, 0)  # 白色

        # 在指定位置添加文字
        draw.text((30, image.size[1]//2), text, font=font, fill=color)

        # 保存修改后的图片
        image.save('resutl.jpg')
        image.show()
