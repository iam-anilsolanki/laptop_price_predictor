# 💻 Laptop Price Predictor

A Streamlit web app to predict laptop prices based on specifications like processor, RAM, storage, weight, and display size. The model is trained on real-world laptop datasets and uses a Random Forest Regressor for accurate predictions.

## 🚀 Features

- User-friendly UI with dropdowns and number inputs
- Predicts laptop prices in INR
- Dark-themed interface with soft purple styling
- Model training and preprocessing cached for faster performance

## 📁 Dataset

The dataset must be placed in the `dataset/` folder as `laptops_train.csv`.  
Ensure the CSV has columns like:

## 🧠 ML Model

- **Model:** RandomForestRegressor (scikit-learn)
- **Encoding:** LabelEncoder for categorical fields
- **Framework:** Streamlit for UI

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/laptop-price-predictor.git
cd laptop-price-predictor
pip install -r requirements.txt
