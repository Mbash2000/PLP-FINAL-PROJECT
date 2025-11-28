import os
import uuid
from flask import Flask, redirect, render_template, request, jsonify
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from torchvision import transforms
from CNN import CNN  # Import the CNN class

# Load disease and supplement info CSV files
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# Load the PyTorch CNN model
model = CNN(39)    
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
model.eval()

def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')  

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/crop-recommendation')
def crop_recommendation():
    soil_types = ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey']
    crop_types = ['Barley', 'Cotton', 'Ground Nuts', 'Maize', 'Millets', 
                 'Oil seeds', 'Paddy', 'Pulses', 'Sugarcane', 'Tobacco', 'Wheat']
    return render_template('crop_recommendation.html', 
                         soil_types=soil_types,
                         crop_types=crop_types)

@app.route('/weather')
def weather_page():
    return render_template('weather.html')

@app.route('/schemes')
def schemes_page():
    return render_template('schemes.html')

@app.route('/data-visuals')
def data_visuals():
    return render_template('DataVisuals.html')

@app.route('/help-line')
def help_line():
    return render_template('HelpLine.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        try:
            image = request.files['image']
            if not image:
                return "No file uploaded", 400
            ext = os.path.splitext(image.filename)[1]
            # generate a unique filename to avoid collisions
            filename = f"{uuid.uuid4().hex}{ext}"
            upload_folder = 'static/uploads'
            os.makedirs(upload_folder, exist_ok=True)
            file_path = os.path.join(upload_folder, filename)
            image.save(file_path)

            pred = prediction(file_path)
            title = disease_info['disease_name'][pred]
            description = disease_info['description'][pred]
            prevent = disease_info['Possible Steps'][pred]
            image_url = disease_info['image_url'][pred]
            supplement_name = supplement_info['supplement name'][pred]
            supplement_image_url = supplement_info['supplement image'][pred]
            supplement_buy_link = supplement_info['buy link'][pred]

            return render_template('submit.html', title=title, desc=description, prevent=prevent,
                                   image_url=image_url, pred=pred, sname=supplement_name,
                                   simage=supplement_image_url, buy_link=supplement_buy_link)
        except Exception as e:
            return f"An error occurred during processing: {e}", 500
    else:
        return redirect('/')

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', 
                           supplement_image=list(supplement_info['supplement image']),
                           supplement_name=list(supplement_info['supplement name']), 
                           disease=list(disease_info['disease_name']), 
                           buy=list(supplement_info['buy link']))

# Crop recommendation and fertilizer prediction functions from app1.py
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load fertilizer recommendation model and preprocessing
fertilizer_model = pickle.load(open('pkl_files/fertilizer_model.pkl', 'rb'))
fertilizer_scaler = pickle.load(open('pkl_files/fertilizer_scaler.pkl', 'rb'))
soil_encoder = pickle.load(open('pkl_files/soil_encoder.pkl', 'rb'))
crop_encoder = pickle.load(open('pkl_files/crop_encoder.pkl', 'rb'))

# Load crop recommendation model and preprocessing
crop_model = pickle.load(open('pkl_files/crop_model.pkl', 'rb'))
crop_scaler = pickle.load(open('pkl_files/crop_scaler.pkl', 'rb'))

# Manual crop dictionary mapping
crop_dict = {
    0: 'rice', 1: 'maize', 2: 'chickpea', 3: 'kidneybeans',
    4: 'pigeonpeas', 5: 'mothbeans', 6: 'mungbean', 7: 'blackgram',
    8: 'lentil', 9: 'pomegranate', 10: 'banana', 11: 'mango',
    12: 'grapes', 13: 'watermelon', 14: 'muskmelon', 15: 'apple',
    16: 'orange', 17: 'papaya', 18: 'coconut', 19: 'cotton',
    20: 'jute', 21: 'coffee'
}

@app.route('/predict_crop', methods=['POST', 'GET'])
def predict_crop():
    try:
        features = [
            float(request.form['N']),
            float(request.form['P']),
            float(request.form['K']),
            float(request.form['temperature']),
            float(request.form['humidity']),
            float(request.form['ph']),
            float(request.form['rainfall'])
        ]
        
        scaled_features = crop_scaler.transform([features])
        prediction = crop_model.predict(scaled_features)
        crop_name = crop_dict.get(int(prediction[0]), 'Unknown Crop')
        
        return jsonify({
            'success': True,
            'prediction': crop_name
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

fertilizer_dict = {
    1: 'Urea',
    2: 'DAP',
    3: '14-35-14',
    4: '28-28',
    5: '17-17-17',
    6: '20-20',
    7: '10-26-26',
}

@app.route('/predict_fertilizer', methods=['POST', 'GET'])
def predict_fertilizer():
    try:
        features = [
            float(request.form['temperature']),
            float(request.form['humidity']),
            float(request.form['moisture']),
            soil_encoder.transform([request.form['soil_type']])[0],
            crop_encoder.transform([request.form['crop_type']])[0],
            float(request.form['nitrogen']),
            float(request.form['potassium']),
            float(request.form['phosphorous'])
        ]

        scaled_features = fertilizer_scaler.transform([features])
        prediction = fertilizer_model.predict(scaled_features)

        fertilizer_index = int(prediction[0])  # Convert numpy int to python int
        fertilizer_name = fertilizer_dict.get(fertilizer_index, "Unknown Fertilizer")

        return jsonify({
            'success': True,
            'prediction': fertilizer_name
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)
