import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Wczytanie danych
dane = np.genfromtxt('155042-regression.txt', skip_header=1)
X = dane[:, 0:-1]
y = dane[:, -1]

# Inicjalizacja modeli (wersje surowe i poprawnie znormalizowane/dostrojone)
models = {
    'Linear Regression': LinearRegression(),
    'Tree (depth=2)': DecisionTreeRegressor(max_depth=2, random_state=42),
    'KNN (Raw)': KNeighborsRegressor(),
    'KNN (Scaled)': make_pipeline(StandardScaler(), KNeighborsRegressor()),
    'MLP (Raw)': MLPRegressor(max_iter=2000, random_state=42),
    'MLP (Scaled)': make_pipeline(StandardScaler(), MLPRegressor(max_iter=2000, random_state=42)),
    'SVR RBF (Raw)': SVR(kernel='rbf'),
    'SVR RBF (Scaled+Tuned)': make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, epsilon=5.0))
}

wyniki = []
for name, model in models.items():
    model.fit(X, y)
    y_pred = model.predict(X)
    wyniki.append({
        'Model': name,
        'MAE': mean_absolute_error(y, y_pred),
        'R2': r2_score(y, y_pred)
    })

df = pd.DataFrame(wyniki)

# --- WYKRES 1: Osobny wykres dla MAE ---
plt.figure(figsize=(12, 6))
bars = plt.bar(df['Model'], df['MAE'], color='skyblue')
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.ylabel('Wartość MAE (Mniej = Lepiej)', fontsize=12)
plt.title('Wpływ standaryzacji MAE', fontsize=14)
# Dodanie etykiet liczbowych na słupkach
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, round(yval, 1), ha='center', va='bottom')
plt.tight_layout()
plt.show()

# --- WYKRES 2: Osobny wykres dla R^2 ---
plt.figure(figsize=(12, 6))
bars2 = plt.bar(df['Model'], df['R2'], color='lightgreen')
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.ylabel('Wartość R^2', fontsize=12)
plt.title('Wpływ standaryzacji  R^2', fontsize=14)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Zaznaczenie zera
# Dodanie etykiet liczbowych z uwzględnieniem ujemnych wyników
for bar in bars2:
    yval = bar.get_height()
    offset = 0.02 if yval >= 0 else -0.05
    va = 'bottom' if yval >= 0 else 'top'
    plt.text(bar.get_x() + bar.get_width() / 2, yval + offset, round(yval, 2), ha='center', va=va)
plt.tight_layout()
plt.show()
# --- WYZNACZENIE NAJLEPSZEGO MODELU Z TWOJEJ LISTY ---
best_model_name = 'SVR RBF (Scaled+Tuned)'
best_model = models[best_model_name]
y_pred_best = best_model.predict(X)

# --- WYKRES 3: Wartości Faktyczne vs Przewidywane ---
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred_best, alpha=0.6, edgecolors='k', color='dodgerblue', label='Predykcje')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Idealne dopasowanie (y=x)')
plt.xlabel('Faktyczne wartości (y)', fontsize=12)
plt.ylabel('Przewidywane wartości przez model', fontsize=12)
plt.title(f'Jakość predykcji - najlepszy model: {best_model_name}', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# --- WYKRES 4: Wykres reszt (pokazuje błędy dla przypadków) ---
residuals = y - y_pred_best
plt.figure(figsize=(10, 6))
plt.scatter(y, residuals, alpha=0.6, edgecolors='k', color='coral')
plt.axhline(0, color='r', linestyle='--', lw=2, label='Brak błędu')
plt.xlabel('Faktyczne wartości (y)', fontsize=12)
plt.ylabel('Błąd predykcji (Faktyczna - Przewidywana)', fontsize=12)
plt.title(f'Rozkład błędów modelu: {best_model_name}', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()