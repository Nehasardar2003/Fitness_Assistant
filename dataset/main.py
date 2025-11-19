import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
import pickle

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def extract_angles_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    data = []
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_csv = f"{video_name}.csv"
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            angles = [
                calculate_angle([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y],
                                [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y],
                                [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]),
                calculate_angle([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y],
                                [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y],
                                [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]),
            ]
            data.append(angles)
            
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(frame, f"Angles: {angles}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow('Training Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    df = pd.DataFrame(data, columns=['Left Elbow', 'Right Elbow'])
    df.to_csv(output_csv, index=False)
    print(f"Angles saved to {output_csv}")
    return output_csv

def train_knn(csv_file):
    df = pd.read_csv(csv_file)
    X = df.values
    y = np.ones(len(X))
    
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)
    
    model_file = csv_file.replace(".csv", ".pkl")
    with open(model_file, 'wb') as f:
        pickle.dump(knn, f)
    print("KNN model trained and saved")
    return model_file

def live_pose_count(model_file):
    with open(model_file, 'rb') as f:
        knn = pickle.load(f)
    
    cap = cv2.VideoCapture(0)
    count = 0
    state = "down"  # Initial state for counting
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            angles = [
                calculate_angle([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y],
                                [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y],
                                [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]),
                calculate_angle([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y],
                                [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y],
                                [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]),
            ]
            prediction = knn.predict([angles])[0]
            
            if prediction == 1:
                if state == "down" and angles[0] > 160:  # Arms straight, move to "up"
                    state = "up"
                elif state == "up" and angles[0] < 90:  # Arms bent, move to "down" and count
                    count += 1
                    print(f"Count: {count}")
                    state = "down"
            
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(frame, f"Angles: {angles}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"Count: {count}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
        cv2.imshow('Live Pose', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Main program
if __name__ == "__main__":
    video_path = "videos/pullup_1.mp4"  # Set your video file here
    if not os.path.exists(video_path):
        print("Error: Video file not found!")
    else:
        csv_file = extract_angles_from_video(video_path)
        model_file = train_knn(csv_file)
        print("Starting live pose detection...")
        live_pose_count(model_file)
