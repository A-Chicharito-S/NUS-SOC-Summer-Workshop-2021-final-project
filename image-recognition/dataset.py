import cv2
import os
from PIL import Image
import os.path
import glob
import numpy
import settings
import matplotlib.pyplot as plt


def make_image(name, index):
    path = 'data-processed/' + name
    if os.path.exists(path):
        return
    else:
        os.makedirs(path)
        cap = cv2.VideoCapture('data-raw/' + name + '.mp4')
        #  cv2.VideoWriter_fourcc(*'XVID')
        #  cap.get(cv2.CAP_PROP_FPS)
        print('reading from the video...')
        i = 0
        while (cap.isOpened()):
            i = i + 1
            ret, frame = cap.read()
            if ret == True:
                if i < (settings.IMAGE_NUMBER + 1):  # 读取前IMAGE_NUMBER张图片作为训练数据
                    cv2.imwrite(path + '/' + index + str(i) + '.jpg', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()
    print('finished.')


def make_dataset(dict):
    data_vgg_size = []  # 照片数量 x (224 x 224 x 3)
    label = []
    for i in range(len(dict)):
        if i == 0:
            print('beginning to make data...')
        else:
            # make_image(dict[i])
            print('beginning to make data...')

        j = 0
        for jpgfile in glob.glob(r"data-processed/" + dict[i] + "/*.jpg"):
            if j < 1500:
                img = Image.open(jpgfile)
                new_img_vgg = img.resize((224, 224), Image.BILINEAR)
                matrix_vgg = numpy.asarray(new_img_vgg)
                to_list_vgg = matrix_vgg.tolist()
                j = j + 1
                data_vgg_size.append(to_list_vgg)
                label.append(i)
            else:
                break

    data_fin_vgg = numpy.array(data_vgg_size)
    label_fin = numpy.array(label)
    return data_fin_vgg, label_fin


def predict_on_your_own(model, number, category):
    if category == 0:
        img = Image.open('data-processed/strangers/1 (' + str(number) + ').jpg')
    else:
        img = Image.open('data-processed/'+str(settings.name_dict[category])+'/' + str(number) + '.jpg')
    new_img_vgg = img.resize((224, 224), Image.BILINEAR)
    matrix_vgg = numpy.asarray(new_img_vgg)
    matrix_vgg = matrix_vgg.reshape((1, 224, 224, 3))
    # 如果报错可能是这个地方的问题，就注释掉再试试, 还有一种可能就是上面的if else里面的open的路径不对，比如拼接错了，少打了空格啥的
    predict = model.predict(matrix_vgg)
    index = numpy.argmax(predict)
    plt.figure("Image")  # 图像窗口名称
    plt.imshow(img)
    plt.axis('off')  # 关掉坐标轴为 off
    plt.title('image from the '+str(settings.name_dict[category])+' class')  # 图像题目
    plt.show()
    print('we think with probability of {} that this picture is: {}, '
          'and the true label is: {}'.format(predict[index], settings.name_dict[index], settings.name_dict[category]))

def predict_random_image(model, path):
    img = Image.open(path)
    # img = img.transpose(Image.ROTATE_270)
    new_img_vgg = img.resize((224, 224), Image.BILINEAR)
    matrix_vgg = numpy.asarray(new_img_vgg)
    matrix_vgg = matrix_vgg.reshape((1, 224, 224, 3))
    predict = model.predict(matrix_vgg)
    print('all predict results:{}'.format(predict))
    index = numpy.argmax(predict)
    plt.figure("Image")  # 图像窗口名称
    plt.imshow(img)
    plt.axis('off')  # 关掉坐标轴为 off
    plt.title('image')  # 图像题目
    plt.show()
    print('we think with probability of {} that this picture is: {}'.format(predict[index], settings.name_dict[index]))


if __name__ == '__main__':
    """
    data_vgg, label = make_dataset(settings.name_dict)
    numpy.save("data_vgg.npy", data_vgg)
    numpy.save("label.npy", label)
    """
data_vgg_size = []  # 照片数量 x (224 x 224 x 3)
label = []

print('making data')
for jpgfile in glob.glob(r"data-processed/strangers/*.jpg"):
    img = Image.open(jpgfile)
    new_img_vgg = img.resize((224, 224), Image.BILINEAR)
    matrix_vgg = numpy.asarray(new_img_vgg)
    to_list_vgg = matrix_vgg.tolist()
    data_vgg_size.append(to_list_vgg)
    label.append(0)


j = 0
print('making data')
for jpgfile in glob.glob(r"data-processed/syc/*.jpg"):
    if j < 2500:
        img = Image.open(jpgfile)
        new_img_vgg = img.resize((224, 224), Image.BILINEAR)
        matrix_vgg = numpy.asarray(new_img_vgg)
        to_list_vgg = matrix_vgg.tolist()
        j = j + 1
        data_vgg_size.append(to_list_vgg)
        label.append(1)
    else:
        break

j = 0
print('making data')
for jpgfile in glob.glob(r"data-processed/jy/*.jpg"):
    if j < 3000:
        img = Image.open(jpgfile)
        new_img_vgg = img.resize((224, 224), Image.BILINEAR)
        matrix_vgg = numpy.asarray(new_img_vgg)
        to_list_vgg = matrix_vgg.tolist()
        j = j + 1
        data_vgg_size.append(to_list_vgg)
        label.append(2)
    else:
        break

j = 0
print('making data')
for jpgfile in glob.glob(r"data-processed/yqs/*.jpg"):
    if j < 3700:
        img = Image.open(jpgfile)
        new_img_vgg = img.resize((224, 224), Image.BILINEAR)
        matrix_vgg = numpy.asarray(new_img_vgg)
        to_list_vgg = matrix_vgg.tolist()
        j = j + 1
        data_vgg_size.append(to_list_vgg)
        label.append(3)
    else:
        break

data_fin_vgg = numpy.array(data_vgg_size)
label_fin = numpy.array(label)
numpy.save("data_vgg.npy", data_fin_vgg)
numpy.save("label.npy", label_fin)
