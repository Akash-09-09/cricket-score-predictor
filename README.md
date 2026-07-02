# IPL T20 Innings Score Predictor

This repository contains a machine learning project that predicts the final score of an IPL innings in real-time based on the current match state (runs scored, wickets lost, overs bowled, and run rates).

## 📂 Project Structure

- **[train_model.py](file:///C:/Users/hugar/OneDrive/Desktop/Cricket_Score_Predictor/train_model.py)**: The complete training pipeline. It cleans the raw ball-by-ball dataset, engineers advanced features, trains both Random Forest and XGBoost regressors, compares them, and saves the best-performing model.
- **[app.py](file:///C:/Users/hugar/OneDrive/Desktop/Cricket_Score_Predictor/app.py)**: An interactive Streamlit web dashboard to input live match details and view real-time score projections.
- **[requirements.txt](file:///C:/Users/hugar/OneDrive/Desktop/Cricket_Score_Predictor/requirements.txt)**: List of necessary libraries to run the project.
- **[.gitignore](file:///C:/Users/hugar/OneDrive/Desktop/Cricket_Score_Predictor/.gitignore)**: Prevents raw CSV datasets and binary model files from being tracked by Git.

---

## 🧠 Feature Engineering & Models

### 1. Engineered Features
To capture the momentum and pressure of T20 cricket, the model engineers:
- **`current_run_rate`**: Overall scoring rate.
- **`rolling_run_rate`**: Scoring rate over the last 3 overs to capture recent momentum.
- **`pressure_index`**: Calculated as `(wickets_fallen * overs_completed / 20)`. This scales the impact of wickets lost relative to how late in the innings it is.
- **`wickets_remaining`** & **`overs_remaining`**: Simple resource capacity markers.

### 2. Model Selection
The training pipeline fits two models:
1. **Random Forest Regressor**
2. **XGBoost Regressor**

It automatically compares their $R^2$ scores on the test set and exports the superior model to `cricket_model.pkl`.

---

## 🛠️ How to Setup and Run

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model:**
   *Make sure you have `ball_by_ball_ipl.csv` in the project root directory.*
   ```bash
   python train_model.py
   ```
   This will output the column mapping results, train both models, print their performance, and save `cricket_model.pkl` and `teams.pkl`.

3. **Launch the Web App:**
   ```bash
   python -m streamlit run app.py
   ```
