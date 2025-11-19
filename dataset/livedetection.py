import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust speed for clarity

def speak_count(count):
    """Function to announce the count via text-to-speech."""
    engine.say(str(count))
    engine.runAndWait()

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

def live_pose_count(model_file):
    with open(model_file, 'rb') as f:
        knn = pickle.load(f)
    
    cap = cv2.VideoCapture(0)
    count = 0
    state = "down"  # Initial state for counting
    accuracy = 0  # Initialize accuracy
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)  # Mirror the image for better user experience
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Key angles for bicep curl
            left_angle = calculate_angle(
                [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y],
                [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y],
                [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]
            )
            
            right_angle = calculate_angle(
                [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y],
                [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y],
                [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]
            )

            angles = [left_angle, right_angle]
            prediction = knn.predict([angles])[0]
            
            # Calculate accuracy based on expected range (ideal angle for curls: 45-160 degrees)
            accuracy = min(100, max(0, (160 - abs(left_angle - 90)) / 160 * 100))
            
            if prediction == 1:
                if state == "down" and left_angle > 160:  # Arms straight, move to "up"
                    state = "up"
                elif state == "up" and left_angle < 90:  # Arms bent, move to "down" and count
                    count += 1
                    # print(f"Count: {count}")
                    speak_count(count)  # Speak the count
                    state = "down"
            
            # Draw pose landmarks
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Draw key lines with colors based on correctness
            for (a, b, c) in [
                (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
                (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST)
            ]:
                a_pos = (int(landmarks[a].x * frame.shape[1]), int(landmarks[a].y * frame.shape[0]))
                b_pos = (int(landmarks[b].x * frame.shape[1]), int(landmarks[b].y * frame.shape[0]))
                c_pos = (int(landmarks[c].x * frame.shape[1]), int(landmarks[c].y * frame.shape[0]))

                color = (0, 255, 0) if accuracy > 70 else (0, 0, 255)  # Green if good, red if bad
                
                cv2.line(frame, a_pos, b_pos, color, 5)
                cv2.line(frame, b_pos, c_pos, color, 5)
                cv2.circle(frame, a_pos, 7, color, -1)
                cv2.circle(frame, b_pos, 7, color, -1)
                cv2.circle(frame, c_pos, 7, color, -1)

            # Draw count on the screen
            cv2.putText(frame, f"Count: {count}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Draw Accuracy Progress Bar
            bar_x, bar_y, bar_width, bar_height = 20, 100, 20, 300
            fill_height = int((accuracy / 100) * bar_height)
            
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
            cv2.rectangle(frame, (bar_x, bar_y + (bar_height - fill_height)), (bar_x + bar_width, bar_y + bar_height), (0, 255, 0), -1)
            cv2.putText(frame, f"{int(accuracy)}%", (bar_x + 30, bar_y + bar_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show the frame
        cv2.imshow('Live Pose', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Main program
if __name__ == "__main__":
    model_file = "pkl modules/curl_1.pkl"  # Replace with your actual model file
    live_pose_count(model_file)
 