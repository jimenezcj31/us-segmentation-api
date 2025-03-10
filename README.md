{\rtf1\ansi\ansicpg1252\cocoartf2761
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # \uc0\u55356 \u57317  Ultrasound Segmentation API\
\
This FastAPI-based **Ultrasound Segmentation API** provides deep-learning-powered segmentation of ultrasound images using a **U-Net model**. The API allows users to upload an image and receive a **segmented mask** as output.\
\
## \uc0\u55357 \u56960  Features\
\uc0\u9989  Upload an ultrasound image and get a segmentation mask  \
\uc0\u9989  Visualize the segmentation results using `/visualize/`  \
\uc0\u9989  Supports multiple image formats (PNG, JPG, etc.)  \
\uc0\u9989  Deployed using **FastAPI + Render**  \
\
---\
\
## \uc0\u55357 \u56524  API Endpoints\
\
### \uc0\u55357 \u56633  **Upload an Image for Segmentation**\
- **URL:** `POST /predict/`\
- **Description:** Accepts an ultrasound image and returns a segmented mask.\
- **Request Example:**\
  ```bash\
  curl -X 'POST' \\\
    'https://your-api.onrender.com/predict/' \\\
    -H 'accept: application/json' \\\
    -H 'Content-Type: multipart/form-data' \\\
    -F 'file=@example_ultrasound.png'}