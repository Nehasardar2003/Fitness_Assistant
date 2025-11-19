from django.db import models
from django.contrib.auth.models import User
from datetime import date

class Pose(models.Model):
    pose_name = models.CharField(max_length=255)
    description = models.TextField()
    uploaded_file = models.FileField(upload_to='pkl_files/')  # Saves to media/pkl_files/

    def __str__(self):
        return self.pose_name
    

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)  # Link to User model
    phone = models.CharField(max_length=15)
    age = models.IntegerField()
    height = models.CharField(max_length=10)
    weight = models.CharField(max_length=10)
    
    def __str__(self):
        return self.user.username
    

class UserExerciseLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    exercise = models.CharField(max_length=255)
    date = models.DateField(default=date.today)

    def __str__(self):
        return f"{self.user.username} - {self.exercise} ({self.date})"
