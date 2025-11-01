#  Car Damage Detection using Deep Learning

A **Streamlit-based web app** that detects **vehicle damage** from uploaded images using a fine-tuned **ResNet50 CNN model**.  
This project demonstrates how computer vision and deep learning can be applied to automate damage assessment from car photos.

---

##  Project Overview

This project uses a **Convolutional Neural Network (CNN)** architecture **(ResNet50)** to classify car images into different **damage categories**.  
Users can upload a photo of a car, and the app will analyze and predict the level or presence of damage.

**Key Features:**
- Upload car images (`.jpg` or `.png`)
- Display uploaded images instantly
- Predict whether the car is **Front Breakage** or **Front Crushed** or **Front Normal** or **Rear Breakage** or **Rear Crushed** or **Rear Normal**
- Deployed locally using Streamlit
- Dataset Link On which model is trained - https://drive.google.com/drive/folders/1BDtieNPJ8Mu3jka9R9w8mGWLStBseJ7A?usp=sharing
---

## Live Url :- https://car-damage-detection-renz2zepzu7ra9cg43rvti.streamlit.app/

---

##  Tech Stack

| Component | Technology Used |
|------------|----------------|
| **Frontend** | Streamlit |
| **Backend** | Python |
| **Deep Learning Model** | PyTorch + ResNet50 |
| **Image Processing** | OpenCV, PIL, torchvision.transforms |
| **Environment** | Anaconda / Virtualenv |

---

##  Project Structure

```
fastapi/                   # optional just for apis needed to any server
streamlit/
│
├── app.py                  # Streamlit web app entry point
├── model_helper.py         # Model loading and prediction helper
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
└── test/                # Sample images for testing
```

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/MohdAbdulRah/car-damage-detection.git
cd car-damage-detection
```

### 2️⃣ Create & Activate Environment
```bash
conda create -n car-damage python=3.10 -y
conda activate car-damage
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit App
```bash
streamlit run app.py
```

### 5️⃣ Upload an Image
- Choose a `.jpg` or `.png` car image.  
- The model will process and display the **damage prediction** instantly.

---

## Model Details

- **Architecture:** ResNet50 pretrained on ImageNet  
- **Fine-tuning:** Custom classifier layer for car damage detection  
- **Framework:** PyTorch  
- **Input Size:** 224x224 pixels  
- **Training Dataset:** Custom dataset of car images labeled as damaged or not damaged  

---

##  Example Prediction


![car](carimage1.PNG)  


![car](carimage2.PNG)

---

##  Common Issues & Fixes

| Issue | Solution |
|--------|-----------|
| `TypeError: ImageMixin.image() got an unexpected keyword argument 'use_container_width'` | Upgrade Streamlit using `pip install --upgrade streamlit` |
| `RuntimeError: invalid hash value` | Delete corrupted weights in `~/.cache/torch/hub/checkpoints` and re-run |
| `torch.cuda.is_available() is False` | Load model with `map_location=torch.device('cpu')` |

---

##  Future Enhancements
- Multi-class classification (minor / moderate / severe damage)
- Integration with car insurance claim system
- Deployment on AWS or Hugging Face Spaces
- REST API endpoint for mobile app integration

---

##  Author

**Mohd Abdul Rahman**  
AI & Full Stack Developer  
[LinkedIn](https://linkedin.com/in/mohd-abdul-rahman-776479285/) | [GitHub](https://github.com/MohdAbdulRah)

---
