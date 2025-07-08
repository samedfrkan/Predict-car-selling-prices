# 🚗 Car Price Prediction with Regression Models

This project analyzes and predicts car prices using the CarDekho dataset. Multiple regression models are evaluated with and without feature selection.

## 📁 Dataset

- `final.csv`: Cleaned version of the CarDekho dataset.
- Target column: `selling_price`
- Features include: year, km_driven, fuel type, transmission, engine, mileage, etc.

## 🧠 Models Used

- Linear Regression
- Support Vector Regressor (SVR)
- Random Forest Regressor
- Gradient Boosting Regressor
- Voting Regressor (ensemble)

## 🛠️ How to Run

First, install dependencies:

```bash
pip install -r requirements.txt
```

Run the model of your choice:

```bash
python pr_project.py --lr     # Linear Regression
python pr_project.py --svr    # Support Vector Regressor
python pr_project.py --rf     # Random Forest
python pr_project.py --gbr    # Gradient Boosting
python pr_project.py --vr     # Voting Regressor
```

## 🔍 Features

- Feature importance analysis using RandomForest
- Automatic feature selection (with a minimum of 3 features)
- Model performance comparison: with and without feature selection
- Outputs:
  - R² Score
  - MAE / RMSE and % errors
  - Prediction vs True Value plots
  - Residual error distribution
  - Feature importance bar charts
  - Sample predictions

## 📊 Output

All plots are saved in the project root:
- `*_prediction_plot.png`
- `*_residual_plot.png`
- `*_feature_importance.png`

## 📎 Authors

- Samed Furkan DEMİR — 152120201070
- İbrahim Batuhan ACAR — 152120201089
