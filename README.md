# Delivery Time Prediction – Mini Machine Learning Experiment

This project is a small but complete machine learning experiment designed to predict **Zomato-style delivery times** using real-world features such as distance, weather, traffic conditions, and courier experience.  
It follows the structure of a standard ML pipeline and focuses on learning, experimentation, and reflection.
This is the Google Collab link :- https://colab.research.google.com/github/ShivangiP2005/delivery-time-ml/blob/main/notebooks/zomato_delivery.ipynb

---

## 📌 1. Project Goal

Apply ML fundamentals on a small dataset and demonstrate:
- data preprocessing  
- model training  
- visualization  
- interpretation and reflection  

The objective is to predict **Delivery_Time_min**, a continuous numerical value → hence **Regression** is used.

---

## 📂 2. Dataset

- Source: Kaggle delivery dataset  
- File used: `delivery.csv`  
- Size: small, easy to experiment with  
- Columns include:  
  - Distance_km  
  - Weather  
  - Traffic_Level  
  - Time_of_Day  
  - Vehicle_Type  
  - Preparation_Time_min  
  - Courier_Experience_yrs  
  - Delivery_Time_min (target)

---

## ⚙️ 3. Steps Performed

### **Step 1: Load the Dataset**
Used pandas to load the CSV and inspect basic structure.

### **Step 2: Preprocessing**
- Checked for missing values  
- Encoded categorical features using `pd.get_dummies()`  
- Normalized numerical features if required  
- Separated features (`X`) and target (`y`)

### **Step 3: Train–Test Split**
Splitted the dataset using `train_test_split(test_size=0.2)` for fair evaluation.

### **Step 4: Model Building**
Two regression models were trained:
1. **Linear Regression** – baseline model  
2. **Random Forest Regression** – captures complex patterns, improved accuracy  

### **Step 5: Model Evaluation**
Measured performance using:
- Mean Absolute Error (MAE)  
- R² Score  

Random Forest performed significantly better.

### **Step 6: Visualization**
Generated a scatter plot of:
- Actual delivery times (x-axis)  
- Predicted delivery times (y-axis)

This helped visually assess model accuracy and error spread.

---

## 🧠 4. Key Learnings

- Real-world data is messy; preprocessing is essential.  
- Linear models are useful for baseline comparisons but often fail on nonlinear patterns.  
- Ensemble models like Random Forest capture complex interactions more effectively.  
- Visualization provides intuition about where the model struggles (long delivery times).  
- Feature quality directly affects accuracy — richer features could further improve prediction.  

---

## 🏁 5. Conclusion

This mini-experiment demonstrates a full ML workflow using a real delivery dataset. It reflects hands-on learning, curiosity, and a practical understanding of how models behave on realistic data.  
The approach is simple, clear, and easy to extend for more advanced ML applications.

---

## 📦 Tools & Libraries Used
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- VS Code  

## 🚀 Future Improvements
- Add advanced features (rush-hour flag, road density, restaurant delays)  
- Tune Random Forest hyperparameters  
- Experiment with Gradient Boosting models  
- Deploy as a small web app  

