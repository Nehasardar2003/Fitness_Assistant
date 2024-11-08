# from sklearn.svm import SVC
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
# from utils import *
# from demo import *
# import cv2
# import pandas as pd

# # Load the training and testing datasets
# data_train = pd.read_csv("train_angle.csv")
# data_test = pd.read_csv("test_angle.csv")

# # Separate features and target labels from the training data
# X, Y = data_train.iloc[:, :data_train.shape[1] - 1], data_train['target']
# print(Y.tolist())  # Convert to list if needed

# import numpy as np

# def calculate_angle(a, b, c):
#     """
#     Calculate the angle between three points.
    
#     Parameters:
#     a, b, c : tuple
#         Coordinates of the points in the form of (x, y).
    
#     Returns:
#     float
#         Angle in degrees.
#     """
#     # Convert points to numpy arrays
#     a = np.array(a)
#     b = np.array(b)
#     c = np.array(c)

#     # Calculate the vectors
#     ab = a - b
#     bc = c - b

#     # Calculate the angle using the dot product
#     cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
#     angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Clip to avoid numerical errors

#     return np.degrees(angle)  # Convert radians to degrees

# # Example usage
# # point_a, point_b, point_c are defined somewhere in your code
# point_a = (1, 2)  # Replace with actual coordinates
# point_b = (2, 3)
# point_c = (3, 2)

# angle_obtained = calculate_angle(point_a, point_b, point_c)
# print(f"The angle obtained is: {angle_obtained:.2f} degrees")

# def dumbbell_reps_counter():
#     # Ask the user for reps and sets
#     user_reps = int(input("Enter the number of repetitions: "))
#     user_sets = int(input("Enter the number of sets: "))
    
#     total_reps = 0
#     current_reps = 0
#     current_sets = 0
    
#     while True:
#         # Capture frame from video feed
#         # Example: frame = cv2.VideoCapture(video_source).read()
        
#         # For demo, replace with actual frame capturing
#         angle_obtained = calculate_angle(point_a, point_b, point_c, '0')  # Replace with actual angle calculation
        
#         # Check if the user is performing a dumbbell lift
#         if angle_obtained < lower_threshold_angle:  # Angle at bottom position
#             current_reps += 1
#             print(f"Reps: {current_reps}/{user_reps}")
            
#             # Optional: Add a short delay to avoid counting multiple reps in one lift
#             cv2.waitKey(1000)  # Wait for 1 second (adjust as needed)

#         if current_reps >= user_reps:
#             current_sets += 1
#             print(f"Set {current_sets} completed!")
#             current_reps = 0  # Reset rep counter
            
#             # Check if all sets are completed
#             if current_sets >= user_sets:
#                 print("All sets completed! Great job!")
#                 break  # Exit the loop or reset as needed

#         # Update your image display logic...
#         cv2.imshow("Video", image)
        
#         # Break the loop on user request (e.g., 'q' key)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Clean up resources
#     cv2.destroyAllWindows()


# # Main flow of the program
# exercise_type = input("Choose exercise type (yoga/dumbbell): ").strip().lower()

# if exercise_type == "yoga":
#     model = SVC(kernel='rbf', decision_function_shape='ovo', probability=True)
#     model.fit(X, Y)
#     # Load the model and data...
#     correct_feedback(model, video_source=0)  # Call the yoga feedback method
# elif exercise_type == "dumbbell":
#     dumbbell_reps_counter()  # Call the dumbbell repetition counting method
# else:
#     print("Invalid choice! Please choose either 'yoga' or 'dumbbell'.")
# # Train the SVC model with 'rbf' kernel
# # model = SVC(kernel='rbf', decision_function_shape='ovo', probability=True)
# # model.fit(X, Y)

# # Evaluate the model on the test dataset
# # predictions = evaluate(data_test, model, show=True)

# # Create and display a confusion matrix
# # cm = confusion_matrix(data_test['target'], predictions)
# # plt.figure(figsize=(8, 6))
# # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
# # plt.xlabel('Predicted Label')
# # plt.ylabel('True Label')
# # plt.title('Confusion Matrix')
# # plt.show()

# # Real-time pose detection using webcam feed
# print("Starting webcam feed for real-time pose detection...")
# # selected_pose="tree"
# # Modify the 'correct_feedback' function to use the webcam
# # correct_feedback(model, video=0)

# cv2.destroyAllWindows()






















from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import cv2
import pandas as pd
import numpy as np
from utils import *
from Demo import *
from dumbell import *

# Load the training and testing datasets
data_train = pd.read_csv("train_angle.csv")
data_test = pd.read_csv("test_angle.csv")

# Separate features and target labels from the training data
X, Y = data_train.iloc[:, :data_train.shape[1] - 1], data_train['target']
print(Y.tolist())  # Convert to list if needed
model = SVC(kernel='rbf', decision_function_shape='ovo', probability=True)
model.fit(X, Y)

# Evaluate the model on the test dataset
predictions = evaluate(data_test, model, show=True)

# Create and display a confusion matrix
cm = confusion_matrix(data_test['target'], predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

exercise_type = input("Choose exercise type (yoga/dumbbell): ").strip().lower()

if exercise_type == "yoga":
    # model = SVC(kernel='rbf', decision_function_shape='ovo', probability=True)
    # model.fit(X, Y)
    correct_feedback(model, video=0)  # Call the yoga feedback method
elif exercise_type == "dumbbell":
    setc=int(input("Enter no of sets: "))
    rep=int(input("Enter no of rep: "))
    dumbbell_curl_counter(setc,rep)  # Call the dumbbell repetition counting method
else:
    print("Invalid choice! Please choose either 'yoga' or 'dumbbell'.")
