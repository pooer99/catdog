<template>

  <div class = "catdog">

    <el-upload
        class="upload"
        action="https://run.mocky.io/v3/9d059bf9-4660-45f2-925d-ce80ad6c4d15"
        drag multiple
        :headers="headers"
        :auto-upload="false"
        :file-list="fileList"
        :on-change="handleChange"
    >
      <el-icon class="el-icon--upload"><upload-filled /></el-icon>
      <div class="el-upload__text">拖拽或者点击<em>上传图片</em></div>
      <template #tip>
        <div class="el-upload__tip">上传的图片大小不能超过2MB</div>
      </template>

    </el-upload>

    <div slot="footer" class="dialog-footer">
      <el-button type="primary" @click="confirmUpload()">智能识别</el-button>
    </div>

  </div>
</template>

<script>
import { UploadFilled } from '@element-plus/icons-vue'

export default {
  name: "catdog",
  components:{
    UploadFilled
  },
  data() {
    return {
      dialogOfUpload: true,
      fileList: [],
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    }
  },
  methods: {
    handleChange(file, fileList) { //文件数量改变
      this.fileList = fileList;
    },

    confirmUpload() { //确认上传
      var param = new FormData();
      this.fileList.forEach(
          (val, index) => {
            param.append("file", val.raw);
          }
      );

      this.axios({
        method: 'post', //提交数据方式选择不限长度的post
        url: 'http://127.0.0.1:5000/upload', //后端地址 url
        data: param, //提交数据
        headers: { 'Content-Type': 'multipart/form-data' }, //默认的请求头格式为json格式，而此处提交的数据是formdata，所以需要设置请求头
      }).then(responce => {
        var result=responce.data['reslut']
          alert("这是：" + result);
      });

      this.$message({
        message: "上传成功！",
        duration: 1000
      });

      //接收识别信息，显示猫狗
      this.axios({
        method: 'get', //提交数据方式选择不限长度的post
        url: 'http://127.0.0.1:5000/upload', //后端地址 url
      }).then(responce => {
        var result=responce.data
        alert("这是：" + result);
      });

    },
  }

}
</script>

<style scoped>
.catdog{
  width: 600px;
  margin-left: 390px;
}
</style>