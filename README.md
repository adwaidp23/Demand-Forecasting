# 📈 Demand Forecasting System

A production-ready demand forecasting system built with Python, using **ARIMA**, **Prophet**, and **LSTM** models to predict future sales trends with high accuracy.

## 🚀 Overview
This project provides an end-to-end pipeline for forecasting product demand. It includes data preprocessing, exploratory data analysis (EDA), feature engineering, and a Streamlit dashboard for interactive forecasting.

## 🛠 Key Features
- **Exploratory Data Analysis**: Visualize trends, seasonality, and anomalies instantly.
- **Triple-Model Forecasting**: 
    - **ARIMA**: Statistical approach for trend and seasonality.
    - **Prophet**: Robust forecasting for holiday effects and periodicities.
    - **LSTM**: Deep learning sequences for complex, non-linear patterns.
- **Performance Optimized**: Streamlit caching ensures lightning-fast interactions by preventing redundant model training.
- **Production Ready**: Fully containerized with Docker and modular source code.

## 📂 Project Structure
```text
├── app.py                  # Streamlit Dashboard entry point
├── generate_data.py        # Synthetic dataset generator
├── Dockerfile              # Containerization configuration
├── requirements.txt        # Project dependencies
├── data/                   # Dataset storage
├── src/                    # Core modules
│   ├── preprocessing.py    # Data cleaning logic
│   ├── eda_tools.py        # Visualization & metrics
│   ├── features.py         # Feature engineering
│   ├── models.py           # ML & Deep Learning models
│   ├── evaluation.py       # Performance metrics (MAE, RMSE)
│   ├── visualization.py    # Forecasting plots
│   └── logger.py           # Centralized logging
└── logs/                   # Application logs
```

## ⚙️ Quick Start

### 1. Local Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Generate test data
python generate_data.py

# Run the app
streamlit run app.py
```

### 2. Docker Deployment
```bash
# Build the image
docker build -t demand-ai .

# Run the container
docker run -p 8501:8501 demand-ai
```

## 📊 Business Insights
The system automatically generates insights such as:
- Overall trend percentages.
- Peak sales days and months.
- Demand volatility metrics.

## ⚖️ License
This project is open-source and available under the MIT License.
