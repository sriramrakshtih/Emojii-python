import os
import cv2
import keras
import numpy as np
from make_dataset import preprocess_image
from face_features import FaceFeatures
from keras import optimizers
from keras.layers import Dense, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping

BATCH_SIZE = 32


def convolution_stack(input_layer):

    c1 = Conv2D(filters=16, kernel_size=(3, 3), activation="relu",
                padding="same")(input_layer)
    c1 = Conv2D(filters=16, kernel_size=(3, 3), activation="relu",
                padding="same")(c1)
    c1 = MaxPooling2D((2, 2), strides=(2, 2))(c1)

    c2 = Conv2D(filters=64, kernel_size=(3, 3), activation="relu",
                padding="same")(c1)
    c2 = Conv2D(filters=16, kernel_size=(3, 3), activation="relu",
                padding="same")(c2)
    c2 = MaxPooling2D((2, 2), strides=(2, 2))(c2)

    c3 = Conv2D(filters=128, kernel_size=(3, 3), activation="relu",
                padding="same")(c2)
    c3 = Conv2D(filters=16, kernel_size=(3, 3), activation="relu",
                padding="same")(c3)
    c3 = MaxPooling2D((2, 2), strides=(2, 2))(c3)

    c4 = Conv2D(filters=256, kernel_size=(3, 3), activation="relu",
                padding="same")(c3)
    c4 = Conv2D(filters=16, kernel_size=(3, 3), activation="relu",
                padding="same")(c4)
    c4 = MaxPooling2D((2, 2), strides=(2, 2))(c4)

    flattened = Flatten()(c4)

    fc1 = Dense(units=256, activation="relu")(flattened)
    partial_model_output_layer = Dense(units=64, activation="relu")(fc1)

    return partial_model_output_layer


def build_hierarchial_model():

    eyes_input = Input(shape=(400, 80, 1), name="eyes_input")
    eyes_model = convolution_stack(input_layer=eyes_input)

    nose_input = Input(shape=(160, 240, 1), name="nose_input")
    nose_model = convolution_stack(input_layer=nose_input)

    mouth_input = Input(shape=(320, 240, 1), name="mouth_input")
    mouth_model = convolution_stack(input_layer=mouth_input)

    layer_summation = keras.layers.concatenate([eyes_model,
                                                nose_model,
                                                mouth_model])

    fully_connected = Dense(units=32, activation="relu",
                            name="fc1")(layer_summation)
    output_layer = Dense(units=5, activation="softmax",
                         name="fc2")(fully_connected)

    model = Model(inputs=[eyes_input, nose_input, mouth_input],
                  outputs=[output_layer],
                  name="prime_model")

    # print model.summary()
    return model


def load_training_data():

    categories = ["anger", "happiness", "neutral", "sadness", "surprise"]
    tr_data_dir = "training_data"

    exs, nxs, mxs, ys = [], [], [], []
    for cid, catg in enumerate(categories):

        for file_ in os.listdir(os.path.join(tr_data_dir, catg)):
            file_path = os.path.join(tr_data_dir, catg, file_)
            fs = FaceFeatures(cv2.imread(file_path)).build_face_structures()[0]
            eyes, nose, mouth = preprocess_image(face_struct=fs)

            exs.append(eyes.reshape(400, 80, 1))
            nxs.append(nose.reshape(160, 240, 1))
            mxs.append(mouth.reshape(320, 240, 1))
            ys.append(cid)

    return np.array(exs), np.array(nxs), np.array(mxs), ys


if __name__ == "__main__":

    model = build_hierarchial_model()
    model.compile(optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    exs, nxs, mxs, ys = load_training_data()
    model.fit(x=[exs, nxs, mxs],
              y=to_categorical(ys, 5),
              epochs=50,
              batch_size=16,
              validation_split=0.15,
              callbacks=[
        ModelCheckpoint('emotion_detect.model',
                        monitor='val_acc',
                        save_best_only=True)])
