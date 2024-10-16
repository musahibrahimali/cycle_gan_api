from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
import tensorflow as tf
import io
from PIL import Image
import numpy as np
import warnings

warnings.filterwarnings('ignore')

app = FastAPI()

# Load both models once when the app starts
generator_g = tf.keras.models.load_model('model/generator_g.keras')  # CT -> MRI
generator_f = tf.keras.models.load_model('model/generator_f.keras')  # MRI -> CT

def preprocess_image(image_file):
    """Preprocess the image: Resize and normalize it."""
    image = Image.open(image_file).convert("RGB")
    image = image.resize((256, 256))  # Resize to the model's expected input size
    image = np.array(image)
    image = (image / 127.5) - 1  # Normalize to [-1, 1]
    image = np.expand_dims(image, axis=0)
    return image

def postprocess_image(image_tensor):
    """Postprocess the image: Rescale and convert back to an image."""
    image = (image_tensor[0] + 1) * 127.5
    image = np.clip(image, 0, 255).astype(np.uint8)
    return Image.fromarray(image)

@app.get('/')
async def home():
    """Home endpoint with a welcome message."""
    return JSONResponse({
        "message": "Hello and welcome to the CT-MRI Regeneration API"
    })

@app.post("/ct-to-mri/")
async def ct_to_mri(file: UploadFile = File(...)):
    """Run inference to convert CT to MRI."""
    # Preprocess the uploaded image (CT image)
    image = preprocess_image(file.file)

    # Run inference using generator_g (CT -> MRI)
    generated_image = generator_g(image, training=False)

    # Postprocess the result to convert back to an image
    generated_image = postprocess_image(generated_image)

    # Save the image to a bytes buffer to return as a response
    buffer = io.BytesIO()
    generated_image.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")

@app.post("/mri-to-ct/")
async def mri_to_ct(file: UploadFile = File(...)):
    """Run inference to convert MRI to CT."""
    # Preprocess the uploaded image (MRI image)
    image = preprocess_image(file.file)

    # Run inference using generator_f (MRI -> CT)
    generated_image = generator_f(image, training=False)

    # Postprocess the result to convert back to an image
    generated_image = postprocess_image(generated_image)

    # Save the image to a bytes buffer to return as a response
    buffer = io.BytesIO()
    generated_image.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")

# To run the app, use:
# uvicorn app:app --reload

if __name__ == '__main__':
    # start the api
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)