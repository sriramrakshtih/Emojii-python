import numpy as np
import cv2
import os


class FaceFeatures(object):

    def __init__(self, input_image):

        # Image preprocessing
        self.input_image = input_image
        self.gray_image = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2GRAY)

        # Loading classifiers
        self.face_clf = cv2.CascadeClassifier("haarcascade_clfs/frontal_face.xml")
        self.eyes_clf = cv2.CascadeClassifier("haarcascade_clfs/eyes_pair.xml")
        self.nose_clf = cv2.CascadeClassifier("haarcascade_clfs/nose.xml")
        self.lips_clf = cv2.CascadeClassifier("haarcascade_clfs/mouth.xml")

    @staticmethod
    def apply_bb(array, bb):
        x, y, w, h = bb

        return array[y:y + h, x:x + w]

    @staticmethod
    def cacl_centroid(bb):
        x, y, w, h = bb

        return (x + w / 2, y + h / 2)

    @staticmethod
    def calc_dist((x1, y1), (x2, y2)):

        return (x1 - x2)**2 + (y1 - y2)**2

    def not_on_nose(self, nose_bb, mouth_bb):
        cmx, cmy = self.cacl_centroid(mouth_bb)
        x, y, w, h = nose_bb

        if cmx < (x + w) and cmy < (y + h):
            return False

        else:
            return True

    def get_faces(self):

        faces = self.face_clf.detectMultiScale(self.gray_image,
                                               minNeighbors=3,
                                               minSize=(200, 200))

        return faces

    def get_eyes(self, face_bb):
        """ Finds eyes with-in the given face bounding box """

        face_region = self.apply_bb(array=self.gray_image, bb=face_bb)
        eye_pairs = self.eyes_clf.detectMultiScale(face_region)

        # Picks the most probable eye_pair among eye_pairs
        # Usually we'll see false positives with eyebrows,
        # so let's pick the bottom-most bb
        if len(eye_pairs) == 0:
            return None

        elif len(eye_pairs) == 1:
            return eye_pairs[0]

        else:
            bottom_most_idx = np.argmax(eye_pairs[:, 1])  # All y vals
            return eye_pairs[bottom_most_idx]

    def get_nose(self, face_bb, eyes_bb):
        """ Finds nose with-in the given face, below but closest to eyes """

        face_region = self.apply_bb(array=self.gray_image, bb=face_bb)
        noses = self.nose_clf.detectMultiScale(face_region)

        if len(noses) == 0:
            return None

        elif len(noses) == 1:
            return noses[0]

        elif eyes_bb is None:
            return noses[0]

        else:
            x, eyes_y, w, h = eyes_bb
            # Filter noses that are below eyes
            possible_noses = [nose for nose in noses if nose[1] > eyes_y]
            # Select nose that is closest to eyes
            eyes_centroid = self.cacl_centroid(eyes_bb)
            noses_distances = []
            for nose_bb in possible_noses:
                nose_centroid = self.cacl_centroid(nose_bb)
                dist = self.calc_dist(eyes_centroid, nose_centroid)
                noses_distances.append(dist)

            if len(noses_distances) > 0:
                target_nose_idx = np.argmin(np.array(noses_distances))

                return possible_noses[target_nose_idx]

            else:

                return None

    def get_mouth(self, face_bb, nose_bb):
        """ Finds mouth with-in the given face that is closest to nose """

        face_region = self.apply_bb(array=self.gray_image, bb=face_bb)
        mouths = self.lips_clf.detectMultiScale(face_region)

        if len(mouths) == 0:
            return None

        elif len(mouths) == 1:
            return mouths[0]

        elif nose_bb is None:
            return mouths[0]

        else:
            x, nose_y, w, h = nose_bb
            # Filter mouths that are below nose
            possible_mouths = [mouth for mouth in mouths if mouth[1] > nose_y]
            # Filter mouths that doesn't coinside with nose
            possible_mouths = [mouth for mouth in possible_mouths
                               if self.not_on_nose(nose_bb, mouth)]
            # Select mouth that is closest to mouth
            nose_centroid = self.cacl_centroid(nose_bb)
            mouths_distances = []
            for mouth_bb in possible_mouths:
                mouth_centroid = self.cacl_centroid(mouth_bb)
                dist = self.calc_dist(nose_centroid, mouth_centroid)
                mouths_distances.append(dist)

            if len(mouths_distances) > 0:
                target_mouth_idx = np.argmin(np.array(mouths_distances))

                return possible_mouths[target_mouth_idx]

            else:
                return None

    def sanity_check(face_structure):
        """ Implement if necessary """
        pass

    def find_faces(self):

        return self.face_clf.detectMultiScale(self.gray_image)

    def build_face_structures_only(self):

        all_faces = self.find_faces()

        all_face_structures = []
        for face_bb in all_faces:
            face_im = self.apply_bb(array=self.gray_image, bb=face_bb)
            my_face = {
                "face": {
                    "bb": face_bb,
                    "im": face_im
                }
            }
            all_face_structures.append(my_face)

        return all_face_structures

    def build_face_structures(self):

        all_faces = self.find_faces()

        all_face_structures = []
        for face_bb in all_faces:

            face_im = self.apply_bb(array=self.gray_image, bb=face_bb)

            these_eyes = self.get_eyes(face_bb=face_bb)
            if these_eyes is not None:
                eyes_im = self.apply_bb(array=face_im, bb=these_eyes)
            else:
                eyes_im = None

            this_nose = self.get_nose(face_bb=face_bb, eyes_bb=these_eyes)
            if this_nose is not None:
                nose_im = self.apply_bb(array=face_im, bb=this_nose)
            else:
                nose_im = None

            this_mouth = self.get_mouth(face_bb=face_bb, nose_bb=this_nose)
            if this_mouth is not None:
                mouth_im = self.apply_bb(array=face_im, bb=this_mouth)
            else:
                mouth_im = None

            my_face = dict()
            my_face.update({
                "face": {"bb": face_bb,
                         "im": face_im,
                         "im_c": self.apply_bb(self.input_image,
                                               bb=face_bb)},
                "eyes": {"bb": these_eyes,
                         "im": eyes_im},
                "nose": {"bb": this_nose,
                         "im": nose_im},
                "mouth": {"bb": this_mouth,
                          "im": mouth_im}
            })

            all_face_structures.append(my_face)

        return all_face_structures


# Run this file for a demo of this module
if __name__ == "__main__":

    for root, dirs, files in os.walk("training_data"):
        for file_ in files:
            if file_.endswith("png") or \
               file_.endswith("jpg") or \
               file_.endswith("tiff"):

                file_path = os.path.join(root, file_)
                input_image = cv2.imread(file_path)
                print(file_path)

                faces_structs = FaceFeatures(input_image).build_face_structures()
                print("There are %d faces in this image" % len(faces_structs))

                for face_struct in faces_structs:
                    face_bb = face_struct["face"]["bb"]
                    if face_bb is not None:
                        x, y, w, h = face_bb
                        cv2.rectangle(input_image, (x, y),
                                      (x + w, y + h), (255, 0, 0), 2)

                        face_region = input_image[y:y + h, x:x + w]

                    eyes_bb = face_struct["eyes"]["bb"]
                    if eyes_bb is not None:
                        x, y, w, h = eyes_bb
                        cv2.rectangle(face_region, (x, y),
                                      (x + w, y + h), (0, 255, 0), 2)

                    nose_bb = face_struct["nose"]["bb"]
                    if nose_bb is not None:
                        x, y, w, h = nose_bb
                        cv2.rectangle(face_region, (x, y),
                                      (x + w, y + h), (0, 0, 255), 2)

                    mouth_bb = face_struct["mouth"]["bb"]
                    if mouth_bb is not None:
                        x, y, w, h = mouth_bb
                        cv2.rectangle(face_region, (x, y),
                                      (x + w, y + h), (0, 255, 255), 2)

                    cv2.imshow("thisImage", input_image)
                    cv2.waitKey(30)
