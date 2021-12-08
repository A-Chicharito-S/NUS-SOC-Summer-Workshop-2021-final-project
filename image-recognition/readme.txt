运行步骤：
1.首先运行dataset.py，会在data-processed文件夹下生成settings.name_dict里面的对应人名的文件夹
     ---每个文件夹包含settings.IMAGE_NUMBER张图片
     ---调整settings.SELECT_INTERVAL则产生一个在每个文件夹里面以settings.SELECT_INTERVAL为间隔选择一张照片的数据集
     ---对应数据会被存在data_vgg.npy文件里面，标签被存在label.npy里面
2.然后运行vggFaceModel.py进行模型的训练和预测
     ---首次运行先pip install keras-vggface，然后第一次的话下载weights可能需要一段时间
     ---调整模型超参直接在settings里面进行修改，删除FaceRecognition-VGG.hd5然后重新运行vggFaceModel.py即可
     ---vggFaceModel.py中的dataset.predict_on_your_own(custom_vgg_model, number=NUMBER, category=CATEGORY)
        用于用训练好的custom_vgg_model对第CATEGORY个类（目前是第0类：stranger，第1类：syc，第2类：jy）的第NUMBER张照片做预测
注意：
   如果要增加识别的人，
   1.直接在settings.name_dict按照之前的数据格式（key为类别，value为名字）加入，
   2.并在data-raw文件架里上传对应的MP4文件（尽量1min左右，命名为name_dict里面的名字.mp4）
   3.删除data-process文件夹下除strangers以外的所有文件，然后重复以上“运行步骤”中的两步