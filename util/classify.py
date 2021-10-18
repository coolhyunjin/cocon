import cv2
import glob
import imutils
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array



def test(model_path, validation_folder, all_labels):
    validation_set = glob.glob(validation_folder + '/data/*')

    for d in range(len(validation_set)):
        img = cv2.imread(validation_set[d])
        output = imutils.resize(img, width=400)
        image = cv2.resize(img, (96, 96))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        model = load_model(model_path + '/MultiLabelBinarizer')

        proba = model.predict(image)[0]
        idxs = np.argsort(proba)[::-1][:2]

        for (i, j) in enumerate(idxs):
            label = "{}: {:.2f}%".format(list(all_labels)[j], proba[j] * 100)
            cv2.putText(output, label, (10, (i * 30) + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        for (label, p) in zip(list(all_labels), proba):
            print("{}: {:.2f}%".format(label, p * 100))

        cv2.imshow("Output", output)
        cv2.waitKey(0)