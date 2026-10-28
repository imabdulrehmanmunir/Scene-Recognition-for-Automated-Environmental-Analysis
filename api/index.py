from flask import Flask, request, jsonify, send_from_directory
import numpy as np
from PIL import Image
import os

# --- NEW CROSS-PLATFORM IMPORT ---
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf
    tflite = tf.lite

# Set up Flask to serve your HTML file locally
base_dir = os.path.dirname(os.path.dirname(__file__))
public_dir = os.path.join(base_dir, 'public')
app = Flask(__name__, static_folder=public_dir)

# Locate the models folder
scratch_path = os.path.join(base_dir, 'models', 'model_scratch.tflite')
transfer_path = os.path.join(base_dir, 'models', 'model_transfer.tflite')

# Load the TFLite models
scratch_interpreter = tflite.Interpreter(model_path=scratch_path)
scratch_interpreter.allocate_tensors()

transfer_interpreter = tflite.Interpreter(model_path=transfer_path)
transfer_interpreter.allocate_tensors()

class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

def predict_image(interpreter, image_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = np.argmax(output_data[0])
    confidence = float(np.max(output_data[0]))
    
    return class_names[predicted_index], confidence

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    try:
        file = request.files['file']
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((150, 150))
        
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        
        scratch_pred, scratch_conf = predict_image(scratch_interpreter, img_array)
        transfer_pred, transfer_conf = predict_image(transfer_interpreter, img_array)
        
        return jsonify({
            'scratch': {'prediction': scratch_pred.capitalize(), 'confidence': round(scratch_conf * 100, 2)},
            'transfer': {'prediction': transfer_pred.capitalize(), 'confidence': round(transfer_conf * 100, 2)}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- ADDED FOR LOCAL TESTING ---
@app.route('/')
def home():
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    print("🚀 Starting local test server at http://localhost:5000")
    app.run(port=5000, debug=True)