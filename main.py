import cv2
import numpy as np
from keras.models import load_model
from face_features import FaceFeatures
from make_dataset import preprocess_image

emojis_dir = "emojis/"
emo_detect_mdl = load_model("emotion_detection_model/emotion_detect.model")
categories = ["anger", "happiness", "neutral", "sadness", "surprise"]


def get_prediction(input_image):
    faces_structs = FaceFeatures(input_image).build_face_structures()

    for face_struct in faces_structs:
        eyes, nose, mouth = preprocess_image(face_struct=face_struct)
        emotion_proba = emo_detect_mdl.predict([eyes.reshape(1, 400, 80, 1),
                                                nose.reshape(1, 160, 240, 1),
                                                mouth.reshape(1, 320, 240, 1)])
        emotion = categories[np.argmax(emotion_proba)]

        face_struct["emotion"] = emotion

    return faces_structs


def map_emoji(input_image, emoji, face_bb):

    x, y, w, h = face_bb
    emoji_r = cv2.resize(emoji, (w, h), interpolation=cv2.INTER_CUBIC)
    input_image[y: y + h, x: x + w] = emoji_r

    return input_image


if __name__ == "__main__":

    video_stream = cv2.VideoCapture(0)

    while(True):

        _, frame = video_stream.read()
        faces_data = get_prediction(frame)

        input_image = frame
        for face_data in faces_data:

            if face_data["face"]["im_c"] is not None:
                face_bb = face_data["face"]["bb"]
                print("Detected %s" % str(face_data["emotion"]))
                emoji = cv2.imread(emojis_dir + face_data["emotion"] + ".png")

                input_image = map_emoji(input_image, emoji, face_bb)

        cv2.imshow("Emotion Detection Demo", input_image)

        k_in = cv2.waitKey(1)
        if k_in == 27:
            break
