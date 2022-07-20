import os.path
from tensorflow.keras import layers, Sequential, optimizers

# 设置LOG等级为 1 警告 2 错误
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

#2.编写神经网络结构
'''
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