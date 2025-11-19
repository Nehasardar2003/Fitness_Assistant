from django.shortcuts import render,redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from .auth import authentication 
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.views.decorators.cache import cache_control
import os
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import google.generativeai as genai
from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import render
from .models import Pose,Profile,UserExerciseLog  # Import the model
from django.contrib.auth.hashers import make_password
import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3
import time
import threading
from django.utils.timezone import now


def index(request):
    return render(request,'index.html')

def register(request):
    if request.method == "POST":
        first_name = request.POST.get("fname")
        last_name = request.POST.get("lname")
        username = request.POST.get("username")  # Email as username
        password = request.POST.get("password")
        confirm_password = request.POST.get("confirmPassword")
        phone = request.POST.get("phone")
        age = request.POST.get("age")
        height = request.POST.get("height")
        weight = request.POST.get("weight")

        if password != confirm_password:
            messages.error(request, "Passwords do not match!")  # Error message
            return redirect("register")

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username (Email) already exists!")  # Error message
            return redirect("register")

        # Create User
        user = User.objects.create_user(
            username=username,
            email=username,  # Email as username
            password=password,
            first_name=first_name,
            last_name=last_name,
        )

        # Create Profile linked to User
        Profile.objects.create(
            user=user,
            phone=phone,
            age=age,
            height=height,
            weight=weight,
        )

        login(request, user)  # Log in the user after registration
        messages.success(request, "Registration successful! You can now log in.")  # Success message
        return redirect("login")  # Redirect to login page

    return render(request, "register.html")


def user_login(request):
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(username=username, password=password)
        if user is not None:
            login(request, user)
            messages.success(request, "Log In Successful!")
            return redirect("dashboard")
        else:
            messages.error(request, "Invalid Username or Password.")
            return redirect("login")
    return render(request, "login.html", {'action': 'login'})


@login_required
def get_profile(request):
    """Fetches logged-in user's profile data."""
    user = request.user

    # Check if the user has a profile
    try:
        profile = user.profile  # OneToOneField relationship
        data = {
            "success": True,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "username": user.username,  # Using username (email)
            "phone": profile.phone,
            "age": profile.age,
            "height": profile.height,
            "weight": profile.weight,
        }
    except Profile.DoesNotExist:
        data = {"success": False, "error": "Profile not found"}

    return JsonResponse(data)

def log_out(request):
    logout(request)
    messages.success(request, "Logged out successfully.")
    return redirect("/")



import os
import time
import threading
import queue
import cv2
import pickle
import numpy as np
from gtts import gTTS
from playsound import playsound
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Queue for speech messages
speech_queue = queue.Queue()

def speak_message(message):
    """ Add messages to queue for speech synthesis """
    speech_queue.put(message)

def _speech_worker():
    """ Process the speech queue to play messages continuously """
    while True:
        message = speech_queue.get()
        if message is None:  # Exit condition
            break
        
        # Generate speech
        tts = gTTS(text=message, lang='en')
        filename = "temp_speech.mp3"
        tts.save(filename)
        
        # Play sound
        playsound(filename)
        
        # Remove the temp file
        os.remove(filename)
        
        speech_queue.task_done()

# Start the speech thread
speech_thread = threading.Thread(target=_speech_worker, daemon=True)
speech_thread.start()

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

def live_pose_count(model_file, exercise_name):
    if not os.path.exists(model_file):
        print("Error: Model file not found.")
        return

    with open(model_file, 'rb') as f:
        knn = pickle.load(f)

    cap = cv2.VideoCapture(0)
    count = -1
    state = "down"
    accuracy = 0
    message_timer = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]

            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]

            left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

            angles = [left_angle, right_angle]
            prediction = knn.predict([angles])[0]

            accuracy = min(100, max(0, (160 - abs(left_angle - 90)) / 160 * 100))
            pose_color = (0, 255, 0) if accuracy > 70 else (0, 0, 255)
            
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=pose_color, thickness=3, circle_radius=5),
                                      mp_drawing.DrawingSpec(color=pose_color, thickness=3))

            if prediction == 1:
                if state == "down" and left_angle > 160:
                    state = "up"
                elif state == "up" and left_angle < 90:
                    count += 1
                    # print(f"Count: {count}")
                    speak_message(f"{count}, Keep it up!")  
                    message_timer = time.time()
                    state = "down"

            cv2.putText(frame, f"Count: {count}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, exercise_name, (frame.shape[1]//2 - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            bar_x, bar_y, bar_width, bar_height = 20, 100, 20, 300
            fill_height = int((accuracy / 100) * bar_height)
            
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
            cv2.rectangle(frame, (bar_x, bar_y + (bar_height - fill_height)), (bar_x + bar_width, bar_y + bar_height), (0, 255, 0), -1)
            cv2.putText(frame, f"{int(accuracy)}%", (bar_x + 30, bar_y + bar_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if time.time() - message_timer < 1:
                cv2.putText(frame, "Keep it up!!!", (frame.shape[1]//2 - 100, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        cv2.imshow('Live Pose', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()



def start_exercise(request):
    if request.method == "POST":
        if not request.user.is_authenticated:
            return JsonResponse({"status": "error", "message": "User not authenticated."})

        exercise_name = request.POST.get("exercise_name")
        today = now().date()

        try:
            exercise = Pose.objects.get(pose_name=exercise_name)
            model_path = os.path.join(settings.MEDIA_ROOT, str(exercise.uploaded_file))

            # Save selected exercise for the user with today's date
            UserExerciseLog.objects.create(user=request.user, exercise=exercise_name, date=today)

            if os.path.exists(model_path):
                live_pose_count(model_path, exercise_name)
                return JsonResponse({"status": "success", "message": "Exercise Ended !"})
            else:
                return JsonResponse({"status": "error", "message": "Model file not found."})
        except Pose.DoesNotExist:
            return JsonResponse({"status": "error", "message": "Exercise not found."})

    return JsonResponse({"status": "error", "message": "Invalid request."})

def get_user_exercises(request):
    if request.user.is_authenticated:
        today = now().date()
        exercises = UserExerciseLog.objects.filter(user=request.user, date=today).values_list('exercise', flat=True)
        return JsonResponse({"exercises": list(exercises)})
    return JsonResponse({"error": "User not authenticated"}, status=401)

from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from .models import UserExerciseLog

from django.utils.timezone import now, timedelta
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from .models import UserExerciseLog  # Ensure you import the correct model

@login_required
def get_exercises_last_five_days(request):
    user = request.user
    today = now().date()
    start_date = today - timedelta(days=5)

    exercises = UserExerciseLog.objects.filter(user=user, date__gte=start_date).values(
        'date', 'exercise'
    ).order_by('-date')

    grouped_exercises = {}
    for entry in exercises:
        date_str = entry['date'].strftime('%Y-%m-%d')
        if date_str not in grouped_exercises:
            grouped_exercises[date_str] = []
        grouped_exercises[date_str].append(entry['exercise'])

    return JsonResponse({"exercises": grouped_exercises})




@login_required(login_url="login")
@cache_control(no_cache = True, must_revalidate = True, no_store = True)
def dashboard(request):
    """Display the dashboard with available exercises."""
    exercises = Pose.objects.all()
    return render(request, 'dashboard.html', {'exercises': exercises})





# Set up API key for Google Generative AI
os.environ['GOOGLE_API_KEY'] = 'AIzaSyDDw4i32pQfN9gRlRAI5JFEg-hzjzlIpLI'
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

@csrf_exempt  # Remove this if you have CSRF protection properly set up
def chatbot_response(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            message = data.get('message')

            if not message:
                return JsonResponse({'error': 'No message provided'}, status=400)

            # Generate response using Gemini Pro
            model = genai.GenerativeModel('gemini-2.0-flash')

            response = model.generate_content(message + " Give answer in 50 to 100 words.")

            return JsonResponse({'response': response.text})

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=400)


def addpklfiles(request):
    if request.method == 'POST' and request.FILES.get('pose_file'):
        # Get form data
        pose_name = request.POST.get('pose_name')
        description = request.POST.get('description')
        uploaded_file = request.FILES['pose_file']  # Handling file upload

        # Save to database
        pose = Pose(pose_name=pose_name, description=description, uploaded_file=uploaded_file)
        pose.save()

        # Add success message
        messages.success(request, f"File uploaded successfully! Pose Name: {pose_name}")

        return redirect('addpklfiles')  # Redirect to the same page or another view

    return render(request, 'addpklfiles.html')
