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
from sklearn.model_selection import cross_validate, KFold
import warnings

warnings.filterwarnings('ignore')

# 1. Wczytanie danych
dane = np.genfromtxt('155042-regression.txt', skip_header=1)
X = dane[:, 0:-1]
y = dane[:, -1]

# 2. Inicjalizacja modeli
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

# 3. Ewaluacja: Cały zbiór vs 10-krotna Kroswalidacja
wyniki = []

# Konfiguracja kroswalidacji (dobrą praktyką jest przetasowanie danych)
cv_strategy = KFold(n_splits=10, shuffle=True, random_state=42)
scoring_metrics = {'mae': 'neg_mean_absolute_error', 'r2': 'r2'}

for name, model in models.items():
    # A. Wyniki na całym zbiorze (pamięciówka)
    model.fit(X, y)
    y_pred_full = model.predict(X)
    mae_full = mean_absolute_error(y, y_pred_full)
    r2_full = r2_score(y, y_pred_full)
    print(name)
    # B. Wyniki z 10-krotnej kroswalidacji
    cv_results = cross_validate(model, X, y, cv=cv_strategy, scoring=scoring_metrics)

    # Zwrócone MAE jest ujemne (konwencja sklearn), więc mnożymy przez -1
    mae_cv_mean = -cv_results['test_mae'].mean()
    r2_cv_mean = cv_results['test_r2'].mean()

    wyniki.append({
        'Model': name,
        'MAE (Cały zbiór)': mae_full,
        'MAE (10-CV)': mae_cv_mean,
        'R2 (Cały zbiór)': r2_full,
        'R2 (10-CV)': r2_cv_mean
    })

df = pd.DataFrame(wyniki)
print("Tabela wyników:\n", df.to_string(index=False))

# --- WIZUALIZACJA ---
x = np.arange(len(df['Model']))  # pozycje modeli na osi X
width = 0.35  # szerokość słupka

# Wykres M1: Porównanie MAE
fig, ax1 = plt.subplots(figsize=(14, 6))
bars1 = ax1.bar(x - width / 2, df['MAE (Cały zbiór)'], width, label='Cały zbiór (Trening)', color='skyblue')
bars2 = ax1.bar(x + width / 2, df['MAE (10-CV)'], width, label='10-Fold CV (Test)', color='salmon')

ax1.set_ylabel('Wartość MAE (Mniej = Lepiej)', fontsize=12)
ax1.set_title('M1: Porównanie błędu MAE (Cały zbiór vs Kroswalidacja)', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(df['Model'], rotation=45, ha='right', fontsize=11)
ax1.legend()
ax1.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Wykres M2: Porównanie R^2
fig, ax2 = plt.subplots(figsize=(14, 6))
bars3 = ax2.bar(x - width / 2, df['R2 (Cały zbiór)'], width, label='Cały zbiór (Trening)', color='lightgreen')
bars4 = ax2.bar(x + width / 2, df['R2 (10-CV)'], width, label='10-Fold CV (Test)', color='orange')

ax2.set_ylabel('Wartość R^2 (Więcej = Lepiej)', fontsize=12)
ax2.set_title('M2: Porównanie współczynnika R^2 (Cały zbiór vs Kroswalidacja)', fontsize=14)
ax2.set_xticks(x)
ax2.set_xticklabels(df['Model'], rotation=45, ha='right', fontsize=11)
ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Zaznaczenie zera
ax2.legend()
ax2.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# --- DODATKOWA WIZUALIZACJA: MLP vs REGRESJA LINIOWA ---
from sklearn.model_selection import cross_val_predict

# 1. Pobieramy predykcje z kroswalidacji (każdy punkt przewidziany w swoim "foldzie testowym")
# Używamy tej samej strategii cv_strategy (KFold), co wcześniej
y_cv_pred_mlp = cross_val_predict(models['MLP (Scaled)'], X, y, cv=cv_strategy)
y_cv_pred_lr = cross_val_predict(models['Linear Regression'], X, y, cv=cv_strategy)

# 2. Tworzymy wykresy porównawcze
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# Wykres dla MLP (Scaled) - PREKDKCJE Z CV
ax1.scatter(y, y_cv_pred_mlp, alpha=0.5, color='purple', edgecolors='k')
ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
ax1.set_title(f'MLP (Scaled) - Predykcje z 10-CV\n(Tu widać prawdziwy błąd)', fontsize=13)
ax1.set_xlabel('Wartości rzeczywiste')
ax1.set_ylabel('Przewidywania (Out-of-sample)')
ax1.grid(True, alpha=0.3)

# Wykres dla Linear Regression - PREKDKCJE Z CV
ax2.scatter(y, y_cv_pred_lr, alpha=0.5, color='seagreen', edgecolors='k')
ax2.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
ax2.set_title(f'Linear Regression - Predykcje z 10-CV', fontsize=13)
ax2.set_xlabel('Wartości rzeczywiste')
ax2.grid(True, alpha=0.3)

plt.suptitle('Porównanie rzetelnych predykcji (Kroswalidacja)', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()