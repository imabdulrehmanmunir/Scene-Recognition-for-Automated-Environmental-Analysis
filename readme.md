# Scene Recognition for Automated Environmental Analysis
**Deep Learning University Project**

**Author:** Abdulrehman Munir  
**Roll No:** SU92-BSCSM-F23-453  

---

## 📌 Problem Statement
With the exponential increase in visual data from satellites, drones, and user uploads, manually categorizing landscape imagery is highly inefficient. This project develops an automated Deep Learning system to classify natural scenes into six distinct categories: **buildings, forests, glaciers, mountains, seas, and streets**. 

By comparing a custom-built Convolutional Neural Network (CNN) against a Transfer Learning approach (MobileNetV2), this project evaluates the trade-offs between computational resources, training time, and classification accuracy.

## 🛠️ Project Architecture
1. **Model 1: Custom CNN (Scratch)** 
   * Architecture: 3 Convolutional Blocks + Flatten + Dense layers.
   * Strengths: Lightweight, custom-built.
   * Limitations: Prone to overfitting on smaller datasets.
2. **Model 2: Transfer Learning (MobileNetV2)** 
   * Architecture: Frozen MobileNetV2 base + GlobalAveragePooling + Dense softmax layer.
   * Strengths: High accuracy (~90%+), fast convergence, excellent generalization.
3. **Deployment Pipeline:**
   * Models compressed to `.tflite` format for serverless deployment.
   * Backend: Python Flask API using `tflite-runtime`.
   * Frontend: HTML/Tailwind CSS with glassmorphism UI.
   * Hosting: Vercel.

## 🔗 Project Links
* **Live Web App:** [Insert Vercel Link Here]
* **Project Demo Video:** [Insert YouTube Link Here]
* **LinkedIn Post:** [Insert LinkedIn Link Here]
* **Dataset Download:** [Insert Drive Link Here]
* **Trained Models (.h5, .keras):** [Insert Drive Link Here]
* **Full Project Report:** [Insert PDF Drive Link Here]

## 💻 How to Run Locally
1. Clone this repository.
2. Ensure you have Python installed.
3. Install dependencies:
   ```bash
   pip install Flask Pillow numpy tensorflow










































































   