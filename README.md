# 🌸 Flower Disease Detection

An AI/ML-based project that uses deep learning and image classification to identify flower diseases from images.

The project focuses on training a machine learning model using flower image data and using the trained model for disease classification.

## 🚀 Features

- Image-based flower disease classification.
- Deep learning model training.
- Automated image prediction using a trained model.
- Python-based implementation.
- Model development and inference pipeline.

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Deep Learning:** TensorFlow / Keras
- **Image Processing:** OpenCV
- **Model Format:** Keras (`.keras`)

## 📁 Project Structure

```text
Flower-Disease-Detection/
│
├── app.py                  # Application and prediction interface
├── model_train.py          # Model training script
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
├── .gitignore              # Ignored files and folders
│
├── flower_data/            # Training dataset (not included)
└── flower_classifier.keras # Trained model (not included)
```

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/namangambhir07/Flower-Disease-Detection.git
```

### 2. Navigate to the Project

```bash
cd Flower-Disease-Detection
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## 🧠 Model Training

The training script is:

```bash
python model_train.py
```

Place the required flower image dataset in the appropriate directory before starting training.

The trained model should be saved as:

```text
flower_classifier.keras
```

## ▶️ Running the Application

After obtaining the trained model, run:

```bash
python app.py
```

The application uses the trained model to perform flower image classification.

## 📦 Dataset & Model

The following files are excluded from this repository because of their size:

- `flower_data/` – Training dataset.
- `flower_classifier.keras` – Trained deep learning model.

These files must be obtained or generated locally before running the complete application.

## 🔮 Future Improvements

- Improve classification accuracy through model optimization.
- Expand the dataset with additional flower varieties and disease categories.
- Develop a more interactive prediction interface.
- Deploy the application as a web-based AI tool.

## 👨‍💻 Author

**Naman Gambhir**

GitHub: [@namangambhir07](https://github.com/namangambhir07)

---

⭐ If you find this project interesting, consider giving the repository a star!
