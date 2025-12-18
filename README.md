# 🌾 Smart Crop Recommendation System

A **hybrid Machine Learning + Data-Driven web application** that helps farmers and agricultural planners select the **most suitable and profitable crops** based on soil, fertilizer, season, and market prices.

---

## 🚀 Features

- 🔐 **User Authentication** (Signup / Login)
- 🌱 **Similar Crop Recommendation** (data-driven similarity)
- 🧪 **Fertilizer-Based Crop Recommendation**
- 🤖 **ML Crop Type Prediction** (trained pipeline)
- 💰 **Market Price Ranking** (city-wise profitability)
- 📊 **History Tracking** of user queries
- 🎨 **Clean, responsive UI** (Flask + CSS)

---

## 🧠 System Type

This project is a **Hybrid Intelligent System**:

| Module | Technique |
|------|---------|
| Crop Type Prediction | Machine Learning |
| Similar Crops | Cosine Similarity (Data-Driven) |
| Fertilizer Recommendation | Rule / Frequency Based |
| Market Ranking | Data-Driven (Modal Price) |

---

## 🏗️ Project Architecture

```
smart-crop-recommendation/
│
├── backend/
│   ├── app.py
│   ├── recommender.py
│   ├── market_ranking.py
│   ├── train_model.py
│   │
│   ├── data/
│   │   ├── Market_Price_Dataset_2024_2025.csv
│   │   └── karnataka_city_crop_fertilizer_dataset_expanded_with_type.csv
│   │
│   ├── artifacts/
│   │   ├── pipeline.pkl
│   │   └── meta.json
│   │
│   ├── templates/
│   ├── static/
│   └── app.db
│
├── .gitignore
└── README.md
```

---

## 📊 Datasets Used

1. **Crop–Soil–Fertilizer Dataset (Karnataka)**
   - Soil pH, NPK values
   - City, season, fertilizer type
   - Crop name & crop type

2. **Market Price Dataset (2024–2025)**
   - City-wise crop prices
   - Min, Max & Modal price

---

## ⚙️ Technologies Used

- **Backend:** Python, Flask
- **Machine Learning:** Scikit-learn, XGBoost
- **Database:** SQLite (SQLAlchemy)
- **Frontend:** HTML, CSS
- **Version Control:** Git, GitHub

---

## ▶️ How to Run the Project

### 1️⃣ Create virtual environment
```
python -m venv venv
venv\Scripts\activate
```

### 2️⃣ Install dependencies
```
pip install -r requirements.txt
```

### 3️⃣ Train ML model
```
python train_model.py --data data/karnataka_city_crop_fertilizer_dataset_expanded_with_type.csv
```

### 4️⃣ Initialize database
```
flask --app app.py init-db
```

### 5️⃣ Run server
```
python app.py
```

Open in browser:
```
http://127.0.0.1:5000
```

---

## 🎓 Viva Highlights

- Uses **ML pipeline** for prediction
- Uses **market intelligence** for profit ranking
- Clear separation of ML & data-driven logic
- Scalable for real-world deployment

---

## 🔮 Future Enhancements

- Weather API integration
- Mobile application
- Real-time market prices
- Fertilizer dosage recommendation
- Multilingual support

---

## 👨‍🎓 Author

**Shreyas M L**  
3rd Year Engineering Student  
Sambhram Institute of Technology, Bengaluru

---

## 📜 License

This project is for **academic and educational use**.
