import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import random
import sys

# Veriyi yükle
file_path = 'final.csv'
df = pd.read_csv(file_path)

X = df.drop(columns=['Unnamed: 0', 'selling_price'])
y = df['selling_price']

# Train-test ayırımı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature'ları ölçekle
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature importance (tüm feature'larla): sadece Random Forest ile
rf_for_importance = RandomForestRegressor(random_state=42)
rf_for_importance.fit(X_train_scaled, y_train)
importances_all = rf_for_importance.feature_importances_
features_all = X.columns
importance_df_all = pd.DataFrame({'Feature': features_all, 'Importance': importances_all}).sort_values(by='Importance', ascending=False)
print("\n--- Feature Importance (Tüm Özelliklerle) ---")
print(importance_df_all)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df_all)
plt.title("Feature Importance (Tüm Özelliklerle)")
plt.tight_layout()
plt.savefig('feature_importance_full.png')
plt.close()

# Feature selection: Random Forest ile önemli özellikleri seç (en az 3 özellik)
selector = SelectFromModel(rf_for_importance, prefit=True)
selected_support = selector.get_support()
if selected_support.sum() < 3:
    # En önemli 3 özelliği al
    top3_idx = np.argsort(importances_all)[-3:]
    selected_support = np.zeros_like(importances_all, dtype=bool)
    selected_support[top3_idx] = True
X_train_selected = X_train_scaled[:, selected_support]
X_test_selected = X_test_scaled[:, selected_support]
selected_features = X.columns[selected_support]
print("\nSeçilen Özellikler:")
print(list(selected_features))

# Modeller 
models = {
    "lr": LinearRegression(),
    "svr": SVR(),
    "rf": RandomForestRegressor(random_state=42),
    "gbr": GradientBoostingRegressor(random_state=42),
    "vr": VotingRegressor(estimators=[
        ('lr', LinearRegression()),
        ('rf', RandomForestRegressor(random_state=42)),
        ('gbr', GradientBoostingRegressor(random_state=42))
    ])
}

# CLI için argümanlar
parser = argparse.ArgumentParser()
parser.add_argument('--lr', action='store_true')
parser.add_argument('--svr', action='store_true')
parser.add_argument('--rf', action='store_true')
parser.add_argument('--gbr', action='store_true')
parser.add_argument('--vr', action='store_true')
args = parser.parse_args()

# Model inputu terminalden alınıyor
selected_key = [k for k, v in vars(args).items() if v]
if not selected_key:
    print("Lütfen bir model seçin: --lr | --svr | --rf | --gbr | --vr")
    sys.exit(1)
model_key = selected_key[0]
model = models[model_key]

# Modeli hem tüm feature'larla hem de seçilmiş feature'larla eğit
for mode, X_tr, X_te, tag in [("Tüm Feature'lar", X_train_scaled, X_test_scaled, "full"), ("Seçilmiş Feature'lar", X_train_selected, X_test_selected, "selected")]:
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)

    print(f"\n--- {model_key.upper()} ({mode}) ---")
    print("R2 Score:", r2_score(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)

    # Gerçek vs Tahmin Grafiği
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Gerçek Satış Fiyatı")
    plt.ylabel("Tahmin Edilen Satış Fiyatı")
    plt.title(f"Gerçek vs Tahmin ({mode})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{model_key}_{tag}_prediction_plot.png')
    plt.close()

    # Hata Grafiği
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, bins=50, kde=True)
    plt.title(f"Tahmin Hataları ({mode})")
    plt.xlabel("Hata")
    plt.ylabel("Frekans")
    plt.tight_layout()
    plt.savefig(f'{model_key}_{tag}_residual_plot.png')
    plt.close()

    # Feature Importance 
    if tag == "selected" and hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        features = selected_features
        importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
        print(importance_df)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title(f"Feature Importance ({mode})")
        plt.tight_layout()
        plt.savefig(f'{model_key}_{tag}_feature_importance.png')
        plt.close()

# Modelin ilk halinden random 10 örnek 
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
sample_indices = random.sample(range(len(y_test)), 10)
sample_true = y_test.iloc[sample_indices].values
sample_pred = y_pred[sample_indices]
print("\nRastgele 10 örnek (Tüm Feature'larla):")
for i, (t, p) in enumerate(zip(sample_true, sample_pred)):
    print(f"{i+1}. Gerçek: {int(t):,} ₺ | Tahmin: {int(p):,} ₺")
