# **Fashion Category Ensemble Classifier** 👗🧠

A **Flask-based web application** that classifies fashion images into **14 apparel categories** using a **confidence-aware ensemble** of **VGG16** and **ResNet50** CNNs trained on a curated **DeepFashion** subset.

---

## ✨ **Features**

- 🔁 Image classification using a **VGG16 + ResNet50 ensemble**  
- ⚖️ **Confidence-aware ensemble logic** that prefers VGG16 with smart fallback to ResNet50  
- 🖱️ Interactive **drag-and-drop web UI** with preview and progress  
- 📊 Shows **top-K predictions**, confidence scores, and raw model outputs  
- ☁️ Model weights hosted on **Hugging Face** (keeps GitHub repo lightweight & clean)

---

> 🚨 **Important:** Model files are intentionally excluded from GitHub.  
> They are hosted separately on **Hugging Face** — see Step 4 below.

---

## 🛠️ **Setup Instructions**

---

### 1. Clone the Repository 🧾

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/fashion-category-ensemble-classifier.git
cd fashion-category-ensemble-classifier

```

### **2. Create and Activate Virtual Environment**
```bash

python -m venv venv

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# Windows (cmd)
venv\Scripts\activate.bat

# macOS / Linux
source venv/bin/activate
```

### **3. Install Dependencies 📦**
```bash

pip install -r requirements.txt
Minimal dependencies:

text

flask
tensorflow
numpy
pillow
```

### **4. Download Model Weights**

Model weights are hosted on Hugging Face:

👉 https://huggingface.co/BrianBobbyJoe/fashion-category-ensemble-classifier

Download the following files and place them inside the models/ directory:

text

models/
├── deepfashion_vgg16_best_model_50epoch_no_dress.keras
└── deepfashion_resnet50_final_model.keras
Models expect 224×224 RGB images with model-specific preprocessing.

### **5. Run the Application ▶️**
```bash
python app.py
```
Open your browser at:
http://127.0.0.1:5000/


