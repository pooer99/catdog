import os.path
import shutil

import tensorflow as tf

# 设置LOG等级为 1 警告 2 错误
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# 1.数据分类处理

# 创建文件夹
def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

# 获取当前路径
runPath = os.getcwd()
data_dir = runPath + '/data/'
mkdir(data_dir)

# 对train里面的猫狗照片进行分类，分别保存到data文件夹下的不同文件夹

'''
猫: 训练数据： train/cat 共10000张
    校验数据： valid/cat 共是2500张
狗: 训练数据： train/dog 共10000张
    校验数据： valid/dog 共2500张
'''

# 训练素材路径
train_dir = data_dir + 'train/'
mkdir(train_dir)
# 校验素材路径
valid_dir = data_dir + 'valid/'
mkdir(valid_dir)

# 猫:
# 训练素材文件夹
cat_dir = train_dir + 'cat/'
mkdir(cat_dir)
# 校验素材文件夹
cat_val_dir = valid_dir + 'cat/'
mkdir(cat_val_dir)

# 狗:
# 训练素材文件夹
dog_dir = train_dir + 'dog/'
mkdir(dog_dir)
# 校验素材文件夹
dog_val_dir = valid_dir + 'dog/'
mkdir(dog_val_dir)

# 分类图片
# 移动图片到指定文件夹
def move_file(file_path, target_path):
    # 提取文件名
    fileName = os.path.split(file_path)[1]
    # 拼接成目标路径
    target_path += '/' + fileName
    print(target_path)
    if os.path.exists(file_path):
        shutil.move(file_path, target_path)
    else:
        print("文件不存在")

# 分离图片到指定文件夹
def changeFile(filetype):
    for i in range(0,12500,1):
        # 文件名格式：train/cat.i.jpg
        filename = train_dir + filetype + '.' + str(i) + '.jpg'
        print(filename)

        #前10000张作为训练素材 后2500张作为验证素材
        # 训练素材 10000
        if i < 10000:
            move_file(filename, train_dir + filetype)
        else:
        # 测试素材 2500
            move_file(filename, valid_dir + filetype)

# 分离猫
changeFile('cat')
# 分离狗
changeFile('dog')


# 图片增强数据预处理, 使用图片生成器
# 返回训练集和校验集预处
from keras.preprocessing import image

def img_transforms():
    # 创建训练数据预处理模板
    train_datagen = image.ImageDataGenerator(
        # 归一化
        rescale=1. / 255,
        # 随机旋转的范围
        rotation_range=40,
        # 随机宽度偏移量
        width_shift_range=0.2,
        # 随机高度偏移量
        height_shift_range=0.2,
        # 随机错切变换
        shear_range=0.2,
        # 随机缩放范围
        zoom_range=0.2,
        # 随机将一半图像水平翻转
        horizontal_flip=True,
        # 填充模式为最近点填充
        fill_mode='nearest'
    )

    # 导入训练集
    train_generator = train_datagen.flow_from_directory(
        # 训练数据路径
        train_dir,
        # 处理后的图片大小128*128
        target_size=(128, 128),
        # 每次训练导入多少张图片
        batch_size=64,
        # 随机数种子
        seed=7,
        # 随机打乱数据
        shuffle=True,
        # 返回2D的one-hot编码标签
        class_mode='categorical'
    )

    # 创建校验数据预处理模板
    valid_datagen = image.ImageDataGenerator(
        # 归一化
        rescale = 1./255
    )

    # 导入校验集
    valid_generator = valid_datagen.flow_from_directory(
        # 校验数据路径
        valid_dir,
        # 处理后的图片大小128*128
        target_size=(128, 128),
        # 每次训练导入多少张图片
        batch_size=64,
        # 随机数种子
        seed=7,
        # 随机打乱数据
        shuffle=False,
        # 返回2D的one-hot编码标签
        class_mode="categorical"
    )

    return train_generator,valid_generator

# 图像预处理
train_generator, valid_generator = img_transforms()
