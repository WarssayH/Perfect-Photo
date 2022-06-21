# Importing the required dependencies
import cv2                                  # For video rendering
import dlib                                 # For face and landmark detection
from scipy.spatial import distance as dist  # For calculating distances between facial landmarks
from imutils import face_utils              # To get the landmark ids of the left and right eyes, mouth and jaw

class PerfectPicture:
    def __init__(self, EYE_THRESH, SMILE_THRESH):
        # Thresholds
        self.EYE_THRESH = EYE_THRESH
        self.SMILE_THRESH = SMILE_THRESH

        # Initializing the models for facial landmark detection
        self.detector = dlib.get_frontal_face_detector()
        self.landmark_predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    """Calculates the EAR (Eye Aspect Ratio) for a given eye to determine if it is open or not"""
    def calc_EAR(self, eye):
        # Calculate the vertical eye distances
        y1 = dist.euclidean(eye[1], eye[5])
        y2 = dist.euclidean(eye[2], eye[4])

        x = 2 * dist.euclidean(eye[0], eye[3])  # Calculate the horizontal eye distance
        EAR = (y1 + y2) / x                     # Calculate the EAR
        return EAR

    """Calculates the MJWR (Mouth to Jaw Width Ratio) for a given mouth, jaw to determine if it is smiling"""
    def calc_MJWR(self, mouth, jaw):
        mw = dist.euclidean(mouth[0], mouth[6])  # Calculate the mouth width
        jw = dist.euclidean(jaw[3], jaw[13])     # Calculate the jaw width
        MJWR = mw / jw                           # Calculate the MJWR
        return MJWR

    """Renders the eyes, mouth, and jaw facial landmarks onto a face within the webcam viewport"""
    def render_landmarks(self, frame, left_eye, right_eye, mouth, jaw, eyes_open, smiling):
        if eyes_open:
            for index, landmark in enumerate(left_eye):
                cv2.circle(frame, landmark, 2, (0, 255, 0), -1)
                if index < len(left_eye) - 1:
                    cv2.line(frame, landmark, left_eye[index + 1], (0, 255, 0), 1)
                cv2.line(frame, left_eye[0], left_eye[-1], (0, 255, 0), 1)
            for index, landmark in enumerate(right_eye):
                cv2.circle(frame, landmark, 2, (0, 255, 0), -1)
                if index < len(right_eye) - 1:
                    cv2.line(frame, landmark, right_eye[index + 1], (0, 255, 0), 1)
                cv2.line(frame, right_eye[0], right_eye[-1], (0, 255, 0), 1)
        else:
            for index, landmark in enumerate(left_eye):
                cv2.circle(frame, landmark, 2, (0, 0, 255), -1)
                if index < len(left_eye) - 1:
                    cv2.line(frame, landmark, left_eye[index + 1], (0, 0, 255), 1)
                cv2.line(frame, left_eye[0], left_eye[-1], (0, 0, 255), 1)
            for index, landmark in enumerate(right_eye):
                cv2.circle(frame, landmark, 2, (0, 0, 255), -1)
                if index < len(right_eye) - 1:
                    cv2.line(frame, landmark, right_eye[index + 1], (0, 0, 255), 1)
                cv2.line(frame, right_eye[0], right_eye[-1], (0, 0, 255), 1)
        if smiling:
            for landmark in mouth:
                cv2.circle(frame, landmark, 2, (0, 255, 0), -1)
            for index in range(0, 11):
                cv2.line(frame, mouth[index], mouth[index + 1], (0, 255, 0), 1)
            cv2.line(frame, mouth[11], mouth[0], (0, 255, 0), 1)
            for index in range(12, 19):
                cv2.line(frame, mouth[index], mouth[index + 1], (0, 255, 0), 1)
            cv2.line(frame, mouth[19], mouth[12], (0, 255, 0), 1)
            for index, landmark in enumerate(jaw):
                cv2.circle(frame, landmark, 2, (0, 255, 0), -1)
                if index < len(jaw) - 1:
                    cv2.line(frame, landmark, jaw[index + 1], (0, 255, 0), 1)
        else:
            for landmark in mouth:
                cv2.circle(frame, landmark, 2, (0, 0, 255), -1)
            for index in range(0, 11):
                cv2.line(frame, mouth[index], mouth[index + 1], (0, 0, 255), 1)
            cv2.line(frame, mouth[11], mouth[0], (0, 0, 255), 1)
            for index in range(12, 19):
                cv2.line(frame, mouth[index], mouth[index + 1], (0, 0, 255), 1)
            cv2.line(frame, mouth[19], mouth[12], (0, 0, 255), 1)
            for index, landmark in enumerate(jaw):
                cv2.circle(frame, landmark, 2, (0, 0, 255), -1)
                if index < len(jaw) - 1:
                    cv2.line(frame, landmark, jaw[index + 1], (0, 0, 255), 1)

    """Renders the EAR, MJWR, eye and mouth status onto the webcam viewport"""
    def render_analysis(self, frame, left_EAR, right_EAR, MJWR, eyes_open, smiling):
        cv2.putText(frame, "LEye Aspect Ratio: " + str(round(left_EAR, 2)), (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2)
        cv2.putText(frame, "REye Aspect Ratio: " + str(round(right_EAR, 2)), (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2)
        cv2.putText(frame, "Mouth to Jaw Width Ratio: " + str(round(MJWR, 2)), (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2)

        if eyes_open:
            cv2.putText(frame, "Eyes open!", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Eye(s) not open", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
        if smiling:
            cv2.putText(frame, "Smiling!", (10, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Not smiling", (10, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

    def analyze_img(self, frame):
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to gray scale to pass to detector
        faces = self.detector(img_gray)                     # Detect all faces
        for face in faces:
            # Detect landmarks and convert the shape class directly to a list of (x,y) coordinates
            shape = self.landmark_predict(img_gray, face)
            shape = face_utils.shape_to_np(shape)

            # Grab facial landmarks
            (LEye_start, LEye_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
            (REye_start, REye_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
            (mouth_start, mouth_end) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
            (jaw_start, jaw_end) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]

            # Parsing the landmarks list to extract left eye and right eye landmarks
            left_eye = shape[LEye_start: LEye_end]
            right_eye = shape[REye_start: REye_end]
            mouth = shape[mouth_start: mouth_end]
            jaw = shape[jaw_start: jaw_end]

            # Calculate and display face's EAR (Eye Aspect Ratio), MJWR (Mouth Jaw Width Ratio)
            left_EAR = self.calc_EAR(left_eye)
            right_EAR = self.calc_EAR(right_eye)
            MJWR = self.calc_MJWR(mouth, jaw)

            # Determine whether the eyes are open, mouth is smiling based on open eyes, smile thresholds
            eyes_open = False
            smiling = False
            if left_EAR > self.EYE_THRESH and right_EAR > self.EYE_THRESH:  # Both eyes are open
                eyes_open = True
            if MJWR > self.SMILE_THRESH:                                    # Mouth is smiling
                smiling = True

            # Render landmarks, stats onto the face
            self.render_landmarks(frame, left_eye, right_eye, mouth, jaw, eyes_open, smiling)
            self.render_analysis(frame, left_EAR, right_EAR, MJWR, eyes_open, smiling)
        return frame
