import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeRegressor
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
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, round(yval, 1), ha='center', va='bottom')
plt.tight_layout()
plt.show()

# --- WYKRES 2: Osobny wykres dla R^2 ---
plt.figure(figsize=(12, 6))
bars2 = plt.bar(df['Model'], df['R2'], color='lightgreen')
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.ylabel('Wartość R^2', fontsize=12)
plt.title('Wpływ standaryzacji  R^2', fontsize=14)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--') # Zaznaczenie zera
# Dodanie etykiet liczbowych z uwzględnieniem ujemnych wyników
for bar in bars2:
    yval = bar.get_height()
    offset = 0.02 if yval >= 0 else -0.05
    va = 'bottom' if yval >= 0 else 'top'
    plt.text(bar.get_x() + bar.get_width()/2, yval + offset, round(yval, 2), ha='center', va=va)
plt.tight_layout()
plt.show()
# Tworzymy model dedykowany do interpretacji (głębokość 3 to świetny kompromis
# między jakością predykcji a zdolnością człowieka do ogarnięcia całego rysunku)
interpretable_tree = DecisionTreeRegressor(max_depth=2, random_state=42)
interpretable_tree.fit(X, y)

# Przygotowanie nazw cech, jeśli je znasz - podmień listę na np. ['Wiek', 'Wzrost', ...]
# Jeśli nie znamy nazw z pliku txt, nazywamy je po prostu Cecha 0, Cecha 1 itd.
feature_cols = [f"Cecha {i}" for i in range(X.shape[1])]

plt.figure(figsize=(16, 8)) # Duży rozmiar dla czytelności
plot_tree(interpretable_tree,
          feature_names=feature_cols,
          filled=True,      # Kolorowanie węzłów (odcienie odzwierciedlają przewidywaną wartość)
          rounded=True,     # Zaokrąglone rogi węzłów
          fontsize=12,
          precision=2)      # Zaokrąglenie wartości do 2 miejsc po przecinku
plt.title('Interpretacja logiczna', fontsize=16)
plt.tight_layout()
plt.show()

# Opcjonalnie: Jakie cechy są najważniejsze?
print("Ważność cech według drzewa decyzyjnego:")
for name, importance in zip(feature_cols, interpretable_tree.feature_importances_):
    if importance > 0:
        print(f" - {name}: {importance:.3f}")