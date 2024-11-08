import cv2
import numpy as np
import mediapipe as mp
from sklearn.metrics import classification_report, confusion_matrix

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize pose detection
pose = mp_pose.Pose()

# Function to calculate the angle between three points (landmarks)
def calculate_angle(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1.x, landmark1.y, landmark1.z
    x2, y2, _ = landmark2.x, landmark2.y, landmark2.z
    x3, y3, _ = landmark3.x, landmark3.y, landmark3.z

    angle = np.degrees(np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2))

    # Check if the angle is less than zero and adjust it.
    if angle < 0:
        angle += 360

    return angle

# Function to extract pose angles from pose landmarks
def extract_pose_angles(results):
    angles = []
    if results.pose_landmarks is not None:
        landmarks = results.pose_landmarks.landmark

        # Wrist angles
        left_wrist_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                           landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value],
                                           landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value])
        angles.append(left_wrist_angle)
        right_wrist_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value])
        angles.append(right_wrist_angle)

        # Elbow angles
        left_elbow_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                           landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                           landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
        angles.append(left_elbow_angle)
        right_elbow_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])
        angles.append(right_elbow_angle)

        # Shoulder angles
        left_shoulder_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                              landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                              landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
        angles.append(left_shoulder_angle)
        right_shoulder_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                               landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                               landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
        angles.append(right_shoulder_angle)

        # Knee angles
        left_knee_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
        angles.append(left_knee_angle)
        right_knee_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                           landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
        angles.append(right_knee_angle)

        # Ankle angles
        left_ankle_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                           landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],
                                           landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value])
        angles.append(left_ankle_angle)
        right_ankle_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
                                            landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value])
        angles.append(right_ankle_angle)

        # Hip angles
        left_hip_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])
        angles.append(left_hip_angle)
        right_hip_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])
        angles.append(right_hip_angle)

    return angles

# Function for real-time pose prediction using a webcam
def predict_video(model, video=0, show=False):
    cap = cv2.VideoCapture(video)
    
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        if results.pose_landmarks:
            # Extract angles
            list_angles = extract_pose_angles(results)
            # Predict pose
            y = model.predict([list_angles])
            name = str(y[0])

            if show:
                mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                (w, h), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                cv2.rectangle(img, (40, 40), (40 + w, 60), (255, 255, 255), cv2.FILLED)
                cv2.putText(img, name, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
                cv2.imshow("Video", img)
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()

# Evaluation function for testing the model on a dataset
def evaluate(data_test, model, show=False):
    target = data_test["target"].values.tolist()
    predictions = []
    for i in range(len(data_test)):
        tmp = data_test.iloc[i, :-1].values.tolist()
        predictions.append(model.predict([tmp])[0])
    if show:
        print(confusion_matrix(predictions, target))
        print(classification_report(predictions, target))
    return predictions
