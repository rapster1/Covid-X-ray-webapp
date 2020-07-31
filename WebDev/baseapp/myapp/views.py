from django.shortcuts import render
from . import forms as F
from .models import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import numpy as np
from PIL import Image
from numpy import asarray

# Create your views here.

def home(request):
    return render(request, 'home.html')

def model(request):
    if request.method == "POST":
        context = F.ImgForm(request.POST, request.FILES)
        img = context.save()
        
        #This will be your classifaction
        result = None
        #--- YOUR MODEL HERE ---
        FinalModel = tf.keras.models.load_model('final_model_v2.h5')
        
        image = Image.open(img.image.path).resize((299,299))
        image_array = np.asarray(image, dtype=np.float32)[np.newaxis]
        image_array = image_array/255

        prediction = FinalModel.predict(image_array)
        print(prediction)

        pred_list = prediction.tolist()
        flatlist = []
        final_list = []
        categories = ['NORMAL','COVID-19','BACTERIAL PNEUMONIA','VIRAL PNEUMONIA','TUBERCULOSIS']
        for elem in pred_list:
            flatlist.extend(elem)
        for num in flatlist:
            num = round(num, 5)
            final_list.append(num)
        final_pred = {'NORMAL': final_list[0],'COVID-19': final_list[1] ,'BACTERIAL PNEUMONIA': final_list[2],'VIRAL PNEUMONIA':final_list[3], 'TUBERCULOSIS':final_list[4]}
        result = final_pred
        
        max1 = np.argmax(prediction)
        decision = "Prediction: " + str(categories[max1])
        print(decision)

        return render(request, 'model.html', {'context': context, 'img': img, 'result': result, 'decision': decision})
        
    else:
        context = F.ImgForm()
    return render(request, 'model.html', {'context': context})
