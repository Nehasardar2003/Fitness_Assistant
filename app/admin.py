from django.contrib import admin
from app.models import Pose,Profile,UserExerciseLog

# Register your models here.
admin.site.register(Pose)
admin.site.register(Profile)
admin.site.register(UserExerciseLog)