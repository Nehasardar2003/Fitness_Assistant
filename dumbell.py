
#Wihtout voice commands

# import cv2
# import mediapipe as mp
# import numpy as np

# # Function to calculate the angle between three points
# def calculate_angle(a, b, c):
#     a = np.array(a)  # First point (shoulder)
#     b = np.array(b)  # Mid point (elbow)
#     c = np.array(c)  # End point (wrist)

#     radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
#               np.arctan2(a[1] - b[1], a[0] - b[0])
#     angle = np.abs(radians * 180.0 / np.pi)

#     if angle > 180.0:
#         angle = 360 - angle

#     return angle

# # Function to perform dumbbell curl counting
# def dumbbell_curl_counter(set_target, rep_target):
#     # Initialize mediapipe pose detection and drawing utilities
#     mp_drawing = mp.solutions.drawing_utils
#     mp_pose = mp.solutions.pose

#     # Start video capture
#     cap = cv2.VideoCapture(0)

#     # Curl counter variables
#     counter = 0
#     stage = None
#     current_set = 1

#     # Setup mediapipe instance
#     with mp_pose.Pose(min_detection_confidence=0.5, 
#                       min_tracking_confidence=0.5) as pose:
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Convert the image from BGR to RGB
#             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             image.flags.writeable = False  # Improve performance

#             # Process the image to detect pose landmarks
#             results = pose.process(image)

#             # Convert the image back to BGR for OpenCV
#             image.flags.writeable = True
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#             # Extract landmarks
#             try:
#                 landmarks = results.pose_landmarks.landmark

#                 # Get the coordinates of shoulder, elbow, and wrist
#                 shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
#                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
#                 elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
#                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
#                 wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, 
#                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

#                 # Calculate the angle between shoulder, elbow, and wrist
#                 angle = calculate_angle(shoulder, elbow, wrist)

#                 # Display the angle on the video feed
#                 cv2.putText(image, str(int(angle)), 
#                             tuple(np.multiply(elbow, [640, 480]).astype(int)), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, 
#                             cv2.LINE_AA)

#                 # Curl counter logic
#                 if angle > 160:
#                     stage = "down"
#                 if angle < 30 and stage == 'down':
#                     stage = "up"
#                     counter += 1
#                     print(f"Curl count: {counter}")

#                     # Reset reps and increment sets if rep target is reached
#                     if counter >= rep_target:
#                         counter = 0
#                         current_set += 1
#                         print(f"Set {current_set} completed!")

#                         # Stop when the set target is reached
#                         if current_set > set_target:
#                             print("Workout complete!")
#                             break

#             except Exception as e:
#                 # Handle exceptions or missing landmarks
#                 pass

#             # Draw the counter and stage box
#             cv2.rectangle(image, (0, 0), (225, 100), (245, 117, 16), -1)
            
#             # Display repetitions count
#             cv2.putText(image, 'REPS', (15, 12), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, 
#                         cv2.LINE_AA)
#             cv2.putText(image, str(counter), 
#                         (10, 60), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, 
#                         cv2.LINE_AA)
            
#             # Display the current stage (up/down)
#             cv2.putText(image, 'STAGE', (65, 12), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, 
#                         cv2.LINE_AA)
#             cv2.putText(image, stage if stage is not None else '', 
#                         (60, 60), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, 
#                         cv2.LINE_AA)

#             # Display set count
#             cv2.putText(image, 'SETS', (15, 90), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, 
#                         cv2.LINE_AA)
#             cv2.putText(image, f'{current_set}/{set_target}', 
#                         (10, 130), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, 
#                         cv2.LINE_AA)

#             # Render pose landmarks on the image
#             mp_drawing.draw_landmarks(
#                 image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                 mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, 
#                                        circle_radius=2), 
#                 mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, 
#                                        circle_radius=2))

#             # Display the video feed with the visualizations
#             cv2.imshow('Dumbbell Curl Counter', image)

#             # Exit the loop when 'q' is pressed
#             if cv2.waitKey(10) & 0xFF == ord('q'):
#                 break

#     # Release the capture and close windows
#     cap.release()
#     cv2.destroyAllWindows()

# # Main program
# if __name__ == "__main__":
#     # Ask the user for target sets and reps
#     set_target = int(input("Enter the number of sets: "))
#     rep_target = int(input("Enter the number of reps per set: "))

#     # Call the dumbbell curl counter function with user inputs
#     dumbbell_curl_counter(set_target, rep_target)


















#With Voice commands

import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
import time

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Text-to-speech engine setup
engine = pyttsx3.init()

# Lock to prevent multiple voice feedbacks at the same time
lock = threading.Lock()

# Function to handle voice feedback
def speak_feedback(text):
    def run():
        with lock:  # Locking to avoid multiple threads using the engine at the same time
            engine.say(text)
            engine.runAndWait()
    threading.Thread(target=run).start()

# Function to perform dumbbell curl counting
def dumbbell_curl_counter(set_target, rep_target):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(0)

    counter = 0
    stage = None
    current_set = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, 
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                angle = calculate_angle(shoulder, elbow, wrist)
                cv2.putText(image, str(int(angle)), 
                            tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, 
                            cv2.LINE_AA)

                if angle > 160:
                    stage = "down"
                if angle < 35 and stage == 'down':
                    stage = "up"
                    counter += 1
                    print(f"Curl count: {counter}")
                    speak_feedback(f"Curl count: {counter}")

                    if counter >= rep_target:
                        counter = 0
                        current_set += 1
                        print(f"Set {current_set} completed!")
                        speak_feedback(f"Set {current_set} completed!")

                        if current_set > set_target:
                            print("Workout complete!")
                            speak_feedback("Workout complete!")
                            break

            except Exception as e:
                pass

            cv2.rectangle(image, (0, 0), (225, 100), (245, 117, 16), -1)

            cv2.putText(image, 'REPS', (15, 12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, 
                        cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, 
                        cv2.LINE_AA)
            
            cv2.putText(image, 'STAGE', (65, 12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, 
                        cv2.LINE_AA)
            cv2.putText(image, stage if stage is not None else '', 
                        (60, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, 
                        cv2.LINE_AA)

            cv2.putText(image, 'SETS', (15, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, 
                        cv2.LINE_AA)
            cv2.putText(image, f'{current_set}/{set_target}', 
                        (10, 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, 
                        cv2.LINE_AA)

            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, 
                                       circle_radius=2), 
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, 
                                       circle_radius=2))

            cv2.imshow('Dumbbell Curl Counter', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# Main program
if __name__ == "__main__":
    set_target = int(input("Enter the number of sets: "))
    rep_target = int(input("Enter the number of reps per set: "))

    dumbbell_curl_counter(set_target, rep_target)
