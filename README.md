
<div align="center">
<img src="https://github.com/irsyamokta/assets/blob/66763825fb801c9a18ce52fae01d2e88c4f807e8/neuroinsight/logo-white.png" width="30%" height="30%" >
</div>

# Tentang

**NeuroInsight ML Service** adalah backend berbasis **FastAPI** yang menangani proses prediksi klasifikasi tumor otak dari citra MRI. Model yang digunakan adalah **Deep Learning berbasis Convolutional Neural Network (CNN)** yang telah dilatih menggunakan dataset citra MRI otak. CNN dipilih karena kemampuannya mengenali pola visual dan fitur kompleks pada citra medis, sehingga mendukung deteksi tumor dengan akurasi yang tinggi.

Terdapat empat kategori klasifikasi:

- **Glioma**
- **Meningioma**
- **Pituitary**
- **No Tumor**

## Fitur Utama

- **Prediksi Tumor Otak dengan CNN**  
  API menerima input citra MRI dan mengembalikan hasil klasifikasi berupa jenis tumor dan probabilitas.

- **RESTful API Berbasis FastAPI**  
  Mendukung integrasi mudah dengan frontend berbasis React.

- **Optimasi untuk Model TensorFlow**  
  Model TensorFlow dioptimalkan agar dapat diakses cepat melalui endpoint prediksi.

## Teknologi yang Digunakan

- **FastAPI** – framework backend untuk penyediaan REST API.  
- **Uvicorn** – server ASGI untuk menjalankan FastAPI.  
- **TensorFlow** – untuk memuat dan menjalankan model CNN.  
- **Pillow** – untuk pemrosesan gambar (image preprocessing).  
- **NumPy** – untuk manipulasi data numerik.  
- **python-multipart** – untuk menangani upload file melalui form-data.  
- **python-dotenv** – untuk mengelola konfigurasi melalui file `.env`.

## Struktur Direktori

```
├── app
│   └── main.py             # File utama FastAPI
│
├── datasets                # Dataset untuk pelatihan model
│   ├── Testing             # Direktory testing
│   └── Training            # Direktori training
│
├── notebook                # Notebook Jupyter untuk eksperimen & training
│
├── output                  # Model yang sudah dilatih (file.h5)
│
├── .env                    # Konfigurasi environment
├── .gitignore
├── config.py               # Konfigurasi tambahan
├── Procfile                # Untuk deployment (misalnya Railway)
└── requirements.txt        # Dependensi proyek
```

## Download Dataset & Notebook

Dataset MRI otak untuk pelatihan model dapat diunduh melalui Kaggle:  

🔗 [Brain Tumor MRI Dataset – Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

Setelah diunduh, letakkan dataset di folder:

```
/datasets
```
Notebook dapat diunduh melalui Google Colab:  

🔗 [Neuroinsight – Colab](https://colab.research.google.com/drive/1-FwGli5zLyz0wfPZNH7a0XS-14_VyyOh?usp=sharing)

Setelah diunduh, letakkan notebook di folder:

```
/notebook
```

## Endpoint Utama

| Method | Endpoint        | Deskripsi                                    |
|--------|-----------------|----------------------------------------------|
| GET    | `/`             | Mengecek status API (health check).          |
| GET    | `/docs`             | Dokumentasi API          |
| POST   | `/api/v1/predict`      | Mengirim citra MRI (form-data) untuk diprediksi. |

## Instalasi

Berikut langkah-langkah untuk menjalankan **NeuroInsight ML Service** secara lokal:

1. **Clone Repository**

   ```bash
   git clone https://github.com/irsyamokta/neuroinsight-ml.git
   ```

2. **Masuk ke Direktori Proyek**

   ```bash
   cd neuroinsight-ml
   ```

3. **Buat Virtual Environment (Opsional namun Direkomendasikan)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # MacOS/Linux
   venv\Scripts\activate     # Windows
   ```

4. **Instal Dependensi**

   ```bash
   pip install -r requirements.txt
   ```

5. **Buat File .env**

   Contoh isi file `.env`:

   ```env
   ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
   ```
   
5. **Buat konfigurasi**

   Contoh isi file `config.py`:

   ```python
   from dotenv import load_dotenv
   import os
    
   load_dotenv()
    
   def get_allowed_origins():
       origins = os.getenv("ALLOWED_ORIGINS", "")
       return [origin.strip() for origin in origins.split(",") if origin.strip()]
   ```

6. **Jalankan Server**

   ```bash
   uvicorn app.main:app --reload
   ```

7. **Akses API di Browser atau Postman**

   ```
   http://127.0.0.1:8000
   ```

## Author

**Tim NeuroInsight**
