from keras.models import load_model
import dataset
model = load_model('FaceRecognition-VGG.hd5')
path = '.jpg' # 这个地方写image的路径
print('predicting...')
dataset.predict_random_image(model, path=path)
