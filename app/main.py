import io
import numpy as np

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from config import get_allowed_origins
from PIL import Image
from datetime import datetime

app = FastAPI(
    title="Neuroinsight API",
    description="API for Neuroinsight",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model("./output/neuroinsight.h5")
img_size = (150, 150)
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

information = {
    "glioma": {
        "description": "Glioma adalah tumor otak yang berasal dari sel glial, yaitu sel yang berfungsi sebagai penunjang sel saraf di otak dan sumsum tulang belakang. Tumor ini dapat muncul di berbagai bagian otak dan memiliki spektrum keganasan, mulai dari yang bersifat jinak (low-grade) hingga sangat agresif (high-grade). Glioma sering menyebabkan gejala seperti sakit kepala, kejang, gangguan penglihatan, kelemahan tubuh, atau perubahan kognitif, tergantung pada lokasi dan ukuran tumor. Karena sifatnya yang invasif dan bisa menyebar ke jaringan otak sekitarnya, deteksi dini serta penanganan yang tepat sangat penting untuk memperlambat progresi penyakit dan meningkatkan prognosis pasien.",
        "clinical_consideration": "Citra menunjukkan kemungkinan besar glioma (>90%). Disarankan evaluasi lanjutan dengan MRI kontras dan konfirmasi melalui biopsi. Diskusi tim multidisiplin (MDT) direkomendasikan untuk perencanaan terapi.",
    },
    "meningioma": {
        "description": "Meningioma adalah tumor otak yang umumnya bersifat jinak dan tumbuh secara perlahan dari meninges, yaitu membran pelindung yang melapisi otak dan sumsum tulang belakang. Meskipun jarang bersifat ganas, meningioma dapat menekan struktur otak di sekitarnya sehingga menimbulkan gejala seperti sakit kepala, gangguan penglihatan, kejang, atau perubahan perilaku. Tumor ini lebih sering ditemukan pada wanita dan individu lanjut usia, dan dalam banyak kasus ditemukan secara tidak sengaja saat pemeriksaan otak rutin. Ukuran, lokasi, dan gejala yang ditimbulkan menentukan apakah diperlukan tindakan medis seperti operasi atau pemantauan berkala.",
        "clinical_consideration": "Karakteristik lesi konsisten dengan meningioma. Disarankan evaluasi lanjutan dengan MRI kontras dan konfirmasi melalui biopsi. Diskusi tim multidisiplin (MDT) direkomendasikan untuk perencanaan terapi.",
    },
    "notumor": {
        "description": "Tidak ditemukan adanya massa atau lesi abnormal dalam hasil pencitraan otak, yang menunjukkan tidak adanya tanda-tanda tumor intrakranial yang dapat dikenali. Struktur otak tampak normal tanpa adanya pembesaran, perubahan densitas jaringan, atau infiltrasi yang mencurigakan. Meski begitu, jika pasien tetap mengalami gejala neurologis seperti sakit kepala kronis, kejang, atau gangguan sensorik, pemeriksaan lanjutan seperti MRI lanjutan atau evaluasi neurologis mungkin diperlukan untuk menyingkirkan kemungkinan gangguan non-struktural atau fungsional.",
        "clinical_consideration": "Tidak ditemukan tumor pada hasil pemindaian ini. Pemeriksaan lanjutan dapat dilakukan jika terdapat efek massa atau gejala neurologis.",
    },
    "pituitary": {
        "description": "Tumor pituitari adalah pertumbuhan abnormal pada kelenjar pituitari yang terletak di dasar otak dan memiliki peran penting dalam mengatur produksi berbagai hormon dalam tubuh. Sebagian besar tumor pituitari bersifat jinak (adenoma), namun dapat mengganggu keseimbangan hormonal tubuh, baik dengan meningkatkan produksi hormon tertentu (hipersekresi) maupun menekannya (hiposekresi). Gejala yang ditimbulkan bisa bervariasi, mulai dari gangguan penglihatan, sakit kepala, gangguan menstruasi, penurunan libido, hingga gangguan metabolisme, tergantung pada jenis dan ukuran tumor serta hormon yang terlibat.",
        "clinical_consideration": "Lesi pada area pituitari terdeteksi. Perlu pemeriksaan lanjutan berupa MRI sellar dan evaluasi kadar hormon untuk menentukan pengaruh endokrin serta penatalaksanaan selanjutnya.",
    }
}

metadata = {
    "model_version": "v1.0",
    "input_shape": "150x150 RGB",
    "prediction_time": datetime.now().isoformat()
}

@app.get("/")
async def root():
    return "Hello Welcome to Neuroinsight API"

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
