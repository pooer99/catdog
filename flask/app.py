import os
from flask import Flask,request
from flask_cors import CORS

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
    file.save(os.path.dirname(__file__) + '/upload/' + file_name)  # 保存文件

    return {
        'code': 200,
        'messsge': "文件上传成功",
    }

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)


