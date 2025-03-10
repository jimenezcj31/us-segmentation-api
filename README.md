# 🏥 Ultrasound Segmentation API

This FastAPI-based **Ultrasound Segmentation API** provides deep-learning-powered segmentation of ultrasound images using a **U-Net model**. The API allows users to upload an image and receive a **segmented mask** as output.

## 🚀 Features
✅ Upload an ultrasound image and get a segmentation mask  
✅ Visualize the segmentation results using `/visualize/`  
✅ Supports multiple image formats (PNG, JPG, etc.)  
✅ Deployed using **FastAPI + Render**  

---

## 📌 API Endpoints

### 🔹 **Upload an Image for Segmentation**
- **URL:** `POST /predict/`
- **Description:** Accepts an ultrasound image and returns a segmented mask.
- **Request Example:**
  ```bash
  curl -X 'POST' \
    'https://your-api.onrender.com/predict/' \
    -H 'accept: application/json' \
    -H 'Content-Type: multipart/form-data' \
    -F 'file=@example_ultrasound.png'
