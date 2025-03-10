import os
import torch
import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Prevent GUI errors when using Matplotlib in FastAPI
import matplotlib.pyplot as plt

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---------------------- MODEL LOADING ---------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UNet(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        def conv_block(in_channels, out_channels):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                torch.nn.ReLU(),
            )

        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)

        self.pool = torch.nn.MaxPool2d(2, 2)
        self.bottleneck = conv_block(512, 1024)

        self.upconv4 = torch.nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = conv_block(1024, 512)
        self.upconv3 = torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = conv_block(512, 256)
        self.upconv2 = torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = conv_block(256, 128)
        self.upconv1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = conv_block(128, 64)

        self.final_conv = torch.nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.final_conv(dec1)

# ---------------------- MODEL AUTO-DOWNLOAD ---------------------- #
def download_model(url, output_path):
    """Download model file if not present"""
    print(f"📥 Downloading model from {url}...")
    response = requests.get(url, stream=True)
    with open(output_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)
    print(f"✅ Model downloaded to {output_path}")

MODEL_URL = "https://drive.google.com/file/d/1iOA5z7ENnE0W6WXcwgY7AjwzhYoM3A5c/view?usp=drive_link"  # Replace with your actual Google Drive link
MODEL_PATH = "unet_best_model.pth"

# Check if model exists, if not, download it
if not os.path.exists(MODEL_PATH):
    download_model(MODEL_URL, MODEL_PATH)

# Load Model
print(f"🔥 Loading model on {device}")
model = UNet().to(device)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading failed: {e}")

# ---------------------- IMAGE PROCESSING ---------------------- #
transform = A.Compose([
    A.Resize(256, 256),
    ToTensorV2(),
])

# FastAPI App
app = FastAPI()

# Enable CORS for Swagger UI and frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to a specific domain if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store the latest uploaded image and its prediction
latest_image = None
latest_prediction = None

def preprocess_image(image):
    """Preprocess image exactly as during model training."""
    global latest_image
    image = image.convert("RGB")
    latest_image = image  # Store the original image for visualization
    image = np.array(image)

    transformed = transform(image=image)
    image_tensor = transformed["image"].unsqueeze(0).to(device).float()

    print(f"🔍 Image tensor shape: {image_tensor.shape}")
    print(f"🔍 Image tensor dtype: {image_tensor.dtype}")

    return image_tensor

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Receive an image and return the segmentation mask."""
    global latest_prediction
    try:
        image = Image.open(BytesIO(await file.read()))
        image_tensor = preprocess_image(image)

        with torch.no_grad():
            pred = model(image_tensor)
            print(f"🔍 Model raw output: {pred}")

            pred_mask = torch.sigmoid(pred).cpu().numpy().squeeze()
            print(f"✅ Model output after sigmoid: {pred_mask}")

            binary_mask = (pred_mask > 0.5).astype(np.uint8)
            print(f"🔍 Binary mask sum: {binary_mask.sum()}")

            latest_prediction = binary_mask  # Store for visualization

            if binary_mask.sum() == 0:
                print("⚠️ Model output is empty (all zeros). Check model weights or input.")

        return {"prediction": binary_mask.tolist()}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/visualize/")
def visualize():
    """Return an image showing the original and predicted segmentation mask."""
    global latest_image, latest_prediction
    if latest_image is None or latest_prediction is None:
        return JSONResponse(status_code=404, content={"error": "No image has been processed yet."})

    # Create a visualization
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Show original image
    ax[0].imshow(latest_image)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    # Show prediction mask
    ax[1].imshow(latest_prediction, cmap="gray")
    ax[1].set_title("Predicted Mask")
    ax[1].axis("off")

    plt.tight_layout()

    # Save image to a buffer
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    
    return Response(buf.getvalue(), media_type="image/png")

# ---------------------- RUN THE SERVER ---------------------- #
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)