# 🏥 ml-medicine

> Модели машинного обучения для ранней диагностики сердечно-сосудистых заболеваний и рака поджелудочной железы.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-1.x-yellow)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-green)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

---

## 📌 О проекте

Два независимых классификатора, обученных на реальных медицинских данных:

| Модель | Задача | Алгоритм | ROC-AUC |
|--------|--------|-----------|---------|
| 🧬 Рак поджелудочной железы | Обнаружение рака по биомаркерам мочи | CatBoost | **0.9522** |
| ❤️ Сердечно-сосудистые заболевания | Оценка риска ССЗ | XGBoost | **0.7989** |

---

## 🧬 Модель: Рак поджелудочной железы

**Ноутбук:** [`pancreatic-cancer.ipynb`](pancreatic-cancer.ipynb)  
**Датасет:** [Urinary Biomarkers for Pancreatic Cancer](https://www.kaggle.com/datasets/johnjdavisiv/urinary-biomarkers-for-pancreatic-cancer) — 590 пациентов (Kaggle)

### Результаты (5-Fold Cross-Validation)

| Метрика | Результат |
|---------|-----------|
| ROC-AUC | **0.9522** |
| Recall | **0.8894** |
| Precision | 0.7988 |
| F1-Score | 0.8413 |

### Важность признаков
1. `plasma_CA19_9` — онкомаркер крови (47.8%)
2. `LYVE1_ratio` — LYVE1, нормализованный на креатинин (11.7%)
3. `LYVE1 × REG1B` — признак взаимодействия (8.0%)

### Методология
- **Целевая переменная:** `diagnosis == 3` (рак поджелудочной) против остальных классов
- Биомаркеры нормализованы на уровень креатинина в моче — это стандартная клиническая практика для устранения влияния гидратации пациента
- Добавлены признаки взаимодействий (`LYVE1×REG1B`, `REG1B×TFF1`) для захвата нелинейных зависимостей
- Дисбаланс классов обработан через `auto_class_weights='Balanced'`
- `early_stopping_rounds=50` на валидационной выборке; финальная модель дообучена на 100% данных

### Входные признаки

| Признак | Описание |
|---------|----------|
| `age` | Возраст (лет) |
| `sex` | Пол (`'M'` или `'F'`) |
| `plasma_CA19_9` | Онкомаркер CA19-9 в крови (Ед/мл). Может быть `NaN`. |
| `creatinine` | Креатинин в моче (мг/дл) |
| `LYVE1` | Биомаркер LYVE1 в моче (нг/мл) |
| `REG1B` | Биомаркер REG1B в моче (нг/мл) |
| `TFF1` | Биомаркер TFF1 в моче (нг/мл) |

### Инференс

```python
import joblib
import pandas as pd

model = joblib.load('pancreatic_cancer_model.joblib')

# df — DataFrame с сырыми данными (признаки выше)
df['sex'] = df['sex'].map({'M': 0, 'F': 1})

df['LYVE1_ratio'] = df['LYVE1'] / df['creatinine']
df['REG1B_ratio'] = df['REG1B'] / df['creatinine']
df['TFF1_ratio']  = df['TFF1']  / df['creatinine']

df['LYVE1_x_REG1B'] = df['LYVE1_ratio'] * df['REG1B_ratio']
df['REG1B_x_TFF1']  = df['REG1B_ratio'] * df['TFF1_ratio']

cols = ['age', 'sex', 'plasma_CA19_9', 'creatinine', 'LYVE1', 'REG1B', 'TFF1',
        'LYVE1_ratio', 'REG1B_ratio', 'TFF1_ratio', 'LYVE1_x_REG1B', 'REG1B_x_TFF1']

# Вероятность рака (0.0 — 1.0)
probabilities = model.predict_proba(df[cols])[:, 1]
```

---

## ❤️ Модель: Сердечно-сосудистые заболевания

**Ноутбук:** [`cardio.ipynb`](cardio.ipynb)  
**Датасет:** [Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) — 70 000 пациентов (Kaggle)

### Результаты (Hold-out выборка, порог = 0.45)

| Метрика | Результат |
|---------|-----------|
| ROC-AUC | **0.7989** |
| Recall | **0.7335** |
| Precision | 0.7274 |
| F1-Score | 0.7304 |
| Accuracy | 0.7319 |

> **Порог 0.45** (вместо стандартного 0.5) выбран намеренно для повышения Recall — в медицинской диагностике пропустить больного пациента опаснее, чем дать ложную тревогу.

### Методология
- Сравнивались XGBoost, LightGBM и CatBoost; финальная модель — XGBoost
- `scale_pos_weight` для компенсации дисбаланса классов
- Добавлен признак `ap_diff` (пульсовое давление = систолическое − диастолическое) как клинически значимый показатель

### Входные признаки

| Признак | Описание | Важно |
|---------|----------|-------|
| `age` | Возраст | **В днях** (лет × 365) |
| `gender` | Пол | 1 = Ж, 2 = М |
| `height` | Рост (см) | |
| `weight` | Вес (кг) | |
| `ap_hi` | Систолическое давление | |
| `ap_lo` | Диастолическое давление | |
| `cholesterol` | Холестерин | 1 — норма, 2 — выше нормы, 3 — высокий |
| `gluc` | Глюкоза | 1 — норма, 2 — выше нормы, 3 — высокая |
| `smoke` | Курение | 0 / 1 |
| `alco` | Алкоголь | 0 / 1 |
| `active` | Физическая активность | 0 / 1 |

### Инференс

```python
import joblib
import pandas as pd

model = joblib.load('cardio_model.joblib')

# df — DataFrame с сырыми данными (признаки выше)
df['ap_diff'] = df['ap_hi'] - df['ap_lo']

if 'id' in df.columns:
    df = df.drop(columns=['id'])

probabilities = model.predict_proba(df)[:, 1]
predictions   = (probabilities > 0.45).astype(int)
```

---

## 🚀 Установка

```bash
pip install pandas numpy catboost xgboost joblib scikit-learn
```

Требуется Python 3.8+.

---

## 📁 Структура репозитория

```
ml-medicine/
├── pancreatic-cancer.ipynb        # EDA, feature engineering, обучение модели
├── pancreatic_cancer_model.joblib
├── cardio.ipynb                   # EDA, сравнение моделей, тюнинг
├── cardio_model.joblib
└── README.md
```
