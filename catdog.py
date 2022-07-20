import os.path
import shutil
import tensorflow as tf
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Sequential, optimizers
from random import randint
import numpy as np

from sklearn.metrics import precision_recall_curve,roc_curve

# 设置LOG等级为 1 警告 2 错误
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# 1.数据分类处理
'''
Written by Luo Changhua
'''
# 创建文件夹
def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

# 获取当前路径
runPath = os.getcwd()
data_dir = runPath + '\data\\'
mkdir(data_dir)

# 对train里面的猫狗照片进行分类，分别保存到data文件夹下的不同文件夹
'''
猫: 训练数据： train/cat 共10000张
    校验数据： valid/cat 共是2500张
狗: 训练数据： train/dog 共10000张
    校验数据： valid/dog 共2500张
'''

# 训练素材路径
train_dir = data_dir + 'train\\'
mkdir(train_dir)
# 校验素材路径
valid_dir = data_dir + 'valid\\'
mkdir(valid_dir)

# 猫:
# 训练素材文件夹
cat_dir = train_dir + 'cat\\'
mkdir(cat_dir)
# 校验素材文件夹
cat_val_dir = valid_dir + 'cat\\'
mkdir(cat_val_dir)

# 狗:
# 训练素材文件夹
dog_dir = train_dir + 'dog\\'
mkdir(dog_dir)
# 校验素材文件夹
dog_val_dir = valid_dir + 'dog\\'
mkdir(dog_val_dir)

# 分类图片
# 移动图片到指定文件夹
def move_file(file_path, target_path):
    # 提取文件名
    fileName = os.path.split(file_path)[1]
    # 拼接成目标路径
    target_path += '\\' + fileName
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

'''
%%%%%%%%%%%%%%%%%%%%%%%%
#这里只需要执行一次即可！！
%%%%%%%%%%%%%%%%%%%%%%%%
'''
# 分离猫
#changeFile('cat')
# 分离狗
#changeFile('dog')


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



#2.编写神经网络结构
'''
Written Guo Qingjun

'''
#CNN 参数依次分别为 样本的 长 宽 深 以及 样本输出个数
def CNN():
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(128, 128, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(2, activation='sigmoid'))
    model.compile(optimizer=optimizers.RMSprop(lr=1e-4),  # 优化器
                  loss='categorical_crossentropy',  # 损失函数
                  metrics=['accuracy']  # 准确率
                  )
    print(model.summary())
    return model


#调用CNN
model = CNN()
#保存模型
modelPath = './model'
mkdir(modelPath)
output_model_file = os.path.join(modelPath,"catdog_CNNweights.h5")



'''
%%%%%%%%%%%%%%%%%%%%%%%%
#从这里到258行，模型训练时跑一次，之后模型保存后不必再跑
%%%%%%%%%%%%%%%%%%%%%%%%
'''
TRAIN_STEP =50

# 设置回调模式
callbacks = [
    tf.keras.callbacks.TensorBoard(modelPath),
    tf.keras.callbacks.ModelCheckpoint(output_model_file,
                                       save_best_only=True,
                                       save_weights_only=True),
    tf.keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)
]

'''
#开始训练
'''

# History = model.fit(
#         train_generator,
#         epochs=TRAIN_STEP,
#         validation_data = valid_generator,
#         callbacks = callbacks
#     )
#
# acc = History.history['accuracy']
# val_acc = History.history['val_accuracy']
# loss = History.history['loss']
# val_loss = History.history['val_loss']
#
# epochs = range(len(acc))
#
# plt.plot(epochs, acc, 'bo', label='Training accuracy')
# plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
# plt.title('Training and validation accuracy')
# plt.legend()#设置图例
#
# plt.figure()
#
# plt.plot(epochs, loss, 'go', label='Training Loss')
# plt.plot(epochs, val_loss, 'g', label='Validation Loss')
# plt.title('Training and validation loss')
# plt.legend()
#
# plt.show()

#模型评估
#输入数据和标签,输出损失和精确度.
test_loss, test_acc = model.evaluate(valid_generator, steps=len(valid_generator), verbose=1)
print('Loss: %.3f' % (test_loss * 100.0))
print('Accuracy: %.3f' % (test_acc * 100.0))

'''
# 预测
# 参数1：要预测的图像cv2格式
# 参数1：模型
# 参数2：特征宽
# 参数3：特征高
# 参数4：特征深度
# 参数5：结果1名称
# 参数6：结果2名称
'''
def predict(cvImg,model,width,height,depth,result1,result2):
    # 加载模型
    model.load_weights('model\catdog_CNNweights.h5')

    # 缩放图像减少检测时间
    img = cv2.resize(cvImg, (width, height))
    # 归一化
    img_arr = img / 255.0
    # 重构成模型需要输入的格式
    img_arr = img_arr.reshape((1, width, height, depth))
    # 输入模型进行预测
    pre = model.predict(img_arr)
    # 打印预测结果
    if pre[0][0] > pre[0][1]:
        print("识别结果是："+result1+"\n概率："+str(pre[0][0]))
    else:
        print("识别结果是："+result2+"\n概率："+str(pre[0][1]))


# 设置样本参数
simpleWight = 128
simpleHeight = 128
simpleDepth = 3

# 设置输出类型
outputNum = 2
result1 = "猫"
result2 = "狗"

#验证一张指定验证图形
img = cv2.imread("data\\test1\\1.jpg")
cv2.imshow('img', img)
predict(img,model,simpleWight,simpleHeight,simpleDepth,result1,result2)
cv2.waitKey(0)

# 随机从测试集中读取文件
test_dir = data_dir + "test1/"
for i in range(10):
    readImg = test_dir + str(randint(0,12499)) + ".jpg"
    img = cv2.imread(readImg)
    predict(img,model,simpleWight,simpleHeight,simpleDepth,result1,result2)
    cv2.imshow("img",img)
    cv2.waitKey(0)

from sklearn.metrics import confusion_matrix
import itertools

predictions = model.predict(x=valid_generator, steps= len(valid_generator), verbose=1)
valid_generator.classes
cm = confusion_matrix(y_true=valid_generator.classes, y_pred=np.argmax(predictions, axis=-1))


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks([-0.5, 1.5], classes)

    print(cm)
    ok_num = 0
    for k in range(cm.shape[0]):
        print(cm[k, k] / np.sum(cm[k, :]))
        ok_num += cm[k, k]

    print(ok_num / np.sum(cm))

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predict label')

cm_plot_labels = ['0-cat', '1-dog']

plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')



def draw_pr(confidence_scores, data_labels):
    plt.figure()
    plt.title('PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid()

    # 精确率，召回率，阈值
    precision, recall, thresholds = precision_recall_curve(data_labels, confidence_scores)

    from sklearn.metrics import average_precision_score
    AP = average_precision_score(data_labels, confidence_scores)  # 计算AP
    plt.plot(recall, precision, label='pr_curve(AP=%0.2f)' % AP)
    plt.legend()
    plt.show()


def draw_roc(confidence_scores, data_labels):
    # 真正率，假正率
    fpr, tpr, thresholds = roc_curve(data_labels, confidence_scores)
    plt.figure()
    plt.grid()
    plt.title('Roc Curve')
    plt.xlabel('FPR')
    plt.ylabel('TPR')

    from sklearn.metrics import auc
    auc = auc(fpr, tpr)  # AUC计算
    plt.plot(fpr, tpr, label='roc_curve(AUC=%0.2f)' % auc)
    plt.legend()
    plt.show()


draw_pr(valid_generator.classes, np.argmax(predictions, axis=-1))
draw_roc(valid_generator.classes, np.argmax(predictions, axis=-1))