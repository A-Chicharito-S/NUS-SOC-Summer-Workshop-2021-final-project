from keras.engine import Model
from keras_vggface.vggface import VGGFace
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import numpy
import keras
import settings
import dataset

data = numpy.load("data_vgg.npy") / 255
label = numpy.load("label.npy")
label = keras.utils.to_categorical(label, len(settings.name_dict))

MODEL_NAME = 'FaceRecognition-VGG.hd5'
train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=settings.TEST_SPLIT)

vgg_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3))
last_layer = vgg_model.get_layer('pool5').output
x = Flatten(name='flatten')(last_layer)
x = Dense(512, activation='relu', name='L6')(x)
x = Dropout(0.5, name='L7')(x)
x = Dense(512, activation='relu', name='L8')(x)
out = Dense(len(settings.name_dict), activation='sigmoid', name='L9')(x)
custom_vgg_model = Model(vgg_model.input, out)

if len(settings.name_dict) < 3:
    loss = 'binary_crossentropy'
else:
    loss = 'categorical_crossentropy'

custom_vgg_model.compile(optimizer=SGD(lr=settings.LR, momentum=settings.MOMENTUM), loss=loss, metrics=['accuracy'])

custom_vgg_model.summary()

savemodel = ModelCheckpoint(MODEL_NAME)
stopmodel = EarlyStopping(min_delta=settings.MIN_DELTA, patience=settings.PATIENCE)
print("Starting training.")

custom_vgg_model.fit(x=train_x, y=train_y, batch_size=settings.VGG_BATCH_SIZE, validation_data=(test_x, test_y),
                     shuffle=True, epochs=settings.VGG_EPOCH, callbacks=[savemodel, stopmodel])

print("Done. Now evaluating.")
loss, acc = custom_vgg_model.evaluate(x=test_x, y=test_y)
print("Test accuracy: %3.2f, loss: %3.2f" % (acc, loss))

# 下面是用来做预测的函数，利用前面的custom_vgg_model对属于CATEGORY类的第NUMBER张图片做预测
NUMBER = 30
CATEGORY = 2
print('here we choose the ' + str(NUMBER) + '-th picture of '
      + str(settings.name_dict[CATEGORY]) + ' to do a illustrative test')
print('predicting...')
dataset.predict_on_your_own(custom_vgg_model, number=NUMBER, category=CATEGORY)
