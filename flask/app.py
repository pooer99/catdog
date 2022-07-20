import os
from flask import Flask,request
from flask_cors import CORS

from Cnn import CNN
import cv2
#导入模型
mymodel = CNN()
mymodel.load_weights('D:\Project\catdog\catdog\model\catdog_CNNweights.h5')
# 设置样本参数
width = 128
height = 128
depth = 3


picpath=''
# 创建Flask实例
app = Flask(__name__,static_folder='uploadimg')
# 全局API配置跨域
cors = CORS(app,origins = '*')
@app.route('/upload', methods = ['POST'])
def uploadPpt():
    file = request.files.get('file')
    if file is None:  # 表示没有发送文件
        return {
            'message': "文件上传失败"
        }
    file_name = file.filename.replace(" ", "")
    print("获取上传文件的名称为[%s]\n" % file_name)
    #注意提前创建uploadppt文件夹哦~
    global  picpath
    picpath=os.path.dirname(__file__) + '\\upload\\' + file_name
    #print(picpath)
    file.save(picpath)  # 保存文件
    Img = cv2.imread(picpath)

    # 缩放图像减少检测时间
    Img = cv2.resize(Img, (width, height))
    # 归一化
    img_arr = Img / 255.0
    # 重构成模型需要输入的格式
    img_arr = img_arr.reshape((1, width, height, depth))
    # 输入模型进行预测
    pre = mymodel.predict(img_arr)
    # 打印预测结果
    if pre[0][0] > pre[0][1]:
        return {
            'code': 200,
            'reslut': "小猫咪"
        }
    else:
        return {
            'code': 200,
            'reslut': "小狗"
        }


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)


