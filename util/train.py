import cv2
import json
import glob
import matplotlib
matplotlib.use("Agg")
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from keras.models import load_model
from tensorflow.keras.optimizers import Adam
from cocon.util.vggnet import SmallerVGGNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.image import img_to_array



def preprocess_data(data, labels, test_ratio):

    data = np.array(data) / 255.
    labels = np.array(labels)

    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(labels)

    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=test_ratio, random_state=42)

    return trainX, testX, trainY, testY



class set_model():


    def parameter(self, train_path, model_path, plot_path, EPOCHS, INIT_LR, BS, all_label):
        self.train_path = train_path
        self.model_path = model_path
        self.plot_path = plot_path
        self.EPOCHS = int(EPOCHS)
        self.INIT_LR = float(INIT_LR)
        self.BS = int(BS)
        self.IMAGE_DIMS = (96, 96, 3)
        self.all_label = all_label


    def batch_generator(self):
        data_lst, label_lst = sorted(glob.glob(self.train_path + '/data/*')), \
                              sorted(glob.glob(self.train_path + '/labels/*'))
        for num in range(len(data_lst) // self.BS):
            data, labels = [], []
            for b in range(self.BS):
               img = cv2.imread(data_lst[num*self.BS + b], cv2.IMREAD_COLOR)
               image = cv2.resize(img, (self.IMAGE_DIMS[1], self.IMAGE_DIMS[0]))
               image = img_to_array(image)
               data.append(image)

               with open(label_lst[num*self.BS + b], encoding='UTF8') as json_file:
                  json_data = json.load(json_file)
                  keys = set(json_data['metadata'].keys())
                  labels.append(list(keys & self.all_label))
            yield data, labels


    def build_model(self):
        resize_and_rescale = tf.keras.Sequential([
           layers.experimental.preprocessing.Resizing(self.IMAGE_DIMS[1], self.IMAGE_DIMS[0]),
           layers.experimental.preprocessing.Rescaling(1./255)
        ])

        data_augmentation = tf.keras.Sequential([
           layers.experimental.preprocessing.RandomFlip("horizontal"),
           layers.experimental.preprocessing.RandomRotation(0.2),
        ])

        model = tf.keras.Sequential([resize_and_rescale, data_augmentation,
                                     layers.Conv2D(16, 3, padding='same', activation='relu'), layers.MaxPooling2D(),
                                     ])

        backbone = SmallerVGGNet.build(width=self.IMAGE_DIMS[1], height=self.IMAGE_DIMS[0], depth=self.IMAGE_DIMS[2],
                                       classes=len(self.all_label), finalAct="sigmoid")

        model.add(backbone)
        opt = Adam(learning_rate=self.INIT_LR, decay=self.INIT_LR / self.EPOCHS)
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
        return model


    def iter_train(self, batch_generator):

        data_lst, label_lst = sorted(glob.glob(self.train_path + '/data/*')), \
                              sorted(glob.glob(self.train_path + '/labels/*'))

        for l in range(len(data_lst) // self.BS):
            data, labels = next(batch_generator)
            trainX, testX, trainY, testY = preprocess_data(data, labels, 0.2)
            model = load_model(self.model_path + '/MultiLabelBinarizer')
            H = model.fit(trainX, trainY, epochs=self.EPOCHS, validation_data=(testX, testY))

            plt.style.use("ggplot")
            plt.figure()
            N = self.EPOCHS
            plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
            plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
            plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
            plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
            plt.title("Training Loss and Accuracy")
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend(loc="upper left")
            plt.show()
            plt.savefig(self.plot_path + '/plot_{}.jpg'.format(l))

            print("[INFO] serializing network...")
            model.save('./cocon/model/MultiLabelBinarizer')

