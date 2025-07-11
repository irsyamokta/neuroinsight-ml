import io
import numpy as np

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from datetime import datetime

app = FastAPI(
    title="Neuroinsight API",
    description="API for Neuroinsight",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model("./output/neuroinsight.h5")
img_size = (150, 150)
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

information = {
    "glioma": {
        "description": "Glioma adalah jenis tumor otak yang berasal dari sel glial dan cenderung agresif.",
        "clinical_consideration": "Citra menunjukkan kemungkinan besar glioma (>90%). Disarankan evaluasi lanjutan dengan MRI kontras dan konfirmasi melalui biopsi. Diskusi tim multidisiplin (MDT) direkomendasikan untuk perencanaan terapi.",
    },
    "meningioma": {
        "description": "Meningioma adalah tumor jinak yang tumbuh di membran pelindung otak dan sumsum tulang belakang.",
        "clinical_consideration": "Karakteristik lesi konsisten dengan meningioma. Disarankan evaluasi lanjutan dengan MRI kontras dan konfirmasi melalui biopsi. Diskusi tim multidisiplin (MDT) direkomendasikan untuk perencanaan terapi.",
    },
    "notumor": {
        "description": "Tidak ditemukan indikasi adanya tumor pada hasil pemindaian ini.",
        "clinical_consideration": "Tidak ditemukan tumor pada hasil pemindaian ini. Pemeriksaan lanjutan dapat dilakukan jika terdapat efek massa atau gejala neurologis.",
    },
    "pituitary": {
        "description": "Tumor pituitari berkembang di kelenjar pituitari dan bisa memengaruhi produksi hormon.",
        "clinical_consideration": "Lesi pada area pituitari terdeteksi. Perlu pemeriksaan lanjutan berupa MRI sellar dan evaluasi kadar hormon untuk menentukan pengaruh endokrin serta penatalaksanaan selanjutnya.",
    }
}

metadata = {
    "model_version": "v1.0",
    "input_shape": "150x150 RGB",
    "prediction_time": datetime.now().isoformat()
}

@app.post("/api/v1/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize(img_size)

    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = round(float(np.max(prediction)) * 100, 2)

    probabilities = {
        class_labels[i]: round(float(prob) * 100, 2)
        for i, prob in enumerate(prediction[0])
    }

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "information": information[predicted_class],
        "probabilities": probabilities,
        "metadata": metadata
    }
