import os
import cv2
import numpy as np
from face_features import FaceFeatures

eyes_dir = "training_data_post_processed/eyes"
nose_dir = "training_data_post_processed/nose"
mouth_dir = "training_data_post_processed/mouth"

categories = ["anger", "happiness", "neutral", "sadness", "surprise"]


def preprocess_image(face_struct):

    eyes = face_struct["eyes"]["im"]
    if eyes is not None:
        eyes_r = cv2.resize(eyes, (400, 80), interpolation=(cv2.INTER_CUBIC))
    else:
        eyes_r = np.zeros((400, 80))

    nose = face_struct["nose"]["im"]
    if nose is not None:
        nose_r = cv2.resize(nose, (240, 160), interpolation=(cv2.INTER_CUBIC))
    else:
        nose_r = np.zeros((240, 160))

    mouth = face_struct["mouth"]["im"]
    if mouth is not None:
        mouth_r = cv2.resize(mouth, (320, 240), interpolation=(cv2.INTER_CUBIC))
    else:
        mouth_r = np.zeros((320, 240))

    return eyes_r, nose_r, mouth_r


def dissect_dataset():
    """ Break given dataset into """

    for root, dirs, files in os.walk("training_data"):
        for file_ in files:
            if file_.endswith("png") or \
                    file_.endswith("JPG") or \
                    file_.endswith("tiff"):

                file_path = os.path.join(root, file_)
                image = cv2.imread(file_path)
                # Since we know all images in our dataset has only one face
                face_struct = FaceFeatures(image).build_face_structures()[0]
                eyes, nose, mouth = preprocess_image(face_struct=face_struct)

                catg = [catg for catg in categories if catg in root].pop()
                if eyes is not None:
                    eyes_path = os.path.join(eyes_dir, catg, file_)
                    cv2.imwrite(eyes_path, eyes)

                if nose is not None:
                    nose_path = os.path.join(nose_dir, catg, file_)
                    cv2.imwrite(nose_path, nose)

                if mouth is not None:
                    mouth_path = os.path.join(mouth_dir, catg, file_)
                    cv2.imwrite(mouth_path, mouth)


if __name__ == "__main__":
    dissect_dataset()
