from django.shortcuts import render
from .utils import predict_image
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
from .train_model import train_model_script

def classify_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            myfile = request.FILES['image']
            fs = FileSystemStorage(location=settings.MEDIA_ROOT)
            filename = fs.save(myfile.name, myfile)
            uploaded_file_url = fs.url(filename)
            
            file_path = os.path.join(settings.MEDIA_ROOT, filename)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            try:
                predicted_class = predict_image(file_path)
            except ValueError as ve:
                print(f"Error during prediction: {ve}")
                return render(request, 'my_classifier/upload.html', {'error': 'Incompatible image shape. Please upload a valid image.'})
            
            return render(request, 'my_classifier/result.html', {'predicted_class': predicted_class, 'uploaded_file_url': uploaded_file_url})
        except Exception as e:
            print(f"Error during file upload or prediction: {e}")
            return render(request, 'my_classifier/upload.html', {'error': str(e)})
    return render(request, 'my_classifier/upload.html')

def train_model(request):
    if request.method == 'POST':
        try:
            train_model_script()
            return render(request, 'my_classifier/train_result.html', {'message': 'Model training completed successfully.'})
        except Exception as e:
            print(f"Error during model training: {e}")
            return render(request, 'my_classifier/train_result.html', {'error': str(e)})
    return render(request, 'my_classifier/train.html')