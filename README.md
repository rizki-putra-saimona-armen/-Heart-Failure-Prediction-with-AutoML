#  Heart Failure Prediction with AutoML

Proyek klasifikasi untuk memprediksi **risiko kematian akibat gagal jantung** (*death event*) menggunakan pendekatan **AutoML** — otomatisasi pemilihan algoritma & hyperparameter tuning terbaik, dibantu library [`jcopml`](https://pypi.org/project/jcopml/) melalui modul `AutoClassifier`.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![AutoML](https://img.shields.io/badge/AutoML-jcopml-purple)
![Status](https://img.shields.io/badge/status-active-success)
![License](https://img.shields.io/badge/license-MIT-green)

---

##  Deskripsi

Proyek ini menunjukkan bagaimana **AutoML** dapat menyederhanakan proses machine learning — daripada mencoba satu per satu algoritma secara manual (Logistic Regression, Random Forest, XGBoost, dll), `AutoClassifier` dari `jcopml` otomatis melatih, membandingkan, dan menuning **banyak algoritma sekaligus**, lalu memberikan rekomendasi model terbaik berdasarkan data.

Studi kasusnya adalah **data medis pasien gagal jantung**, dengan tujuan memprediksi kolom target `DEATH_EVENT` (apakah pasien meninggal akibat gagal jantung dalam periode observasi atau tidak).

Proyek ini cocok sebagai referensi belajar:
- Cara memakai **AutoML** untuk mempercepat eksperimen awal (*rapid prototyping*) sebelum deep-dive ke satu algoritma
- Strategi **dua tahap tuning**: eksplorasi banyak algoritma → fokus & perkuat tuning pada algoritma terbaik
- Analisis **feature importance** pada data medis untuk memahami faktor risiko paling berpengaruh

---

##  Struktur Proyek
```
.
├── autoMl.ipynb                        # Notebook utama
├── data/
│   └── prediksi_gagal_jantung.csv      # Dataset rekam medis pasien
├── model/                              # Output model tersimpan (opsional)
└── README.md
```

> Dataset merupakan data rekam medis dengan kolom target **`DEATH_EVENT`** (biner: 0 = hidup, 1 = meninggal), dengan kombinasi fitur numerik (hasil tes lab) dan fitur biner (kondisi medis).

---

##  Instalasi

1. Clone repository ini:
```bash
git clone https://github.com/username/nama-repo.git
cd nama-repo
```

2. Buat virtual environment (opsional tapi disarankan):
```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

3. Install dependencies:
```bash
pip install numpy pandas scikit-learn matplotlib jupyter
pip install jcopml xgboost
```

---

##  Fitur Dataset

| Tipe Fitur | Kolom |
|---|---|
| **Numerik** | `age`, `creatinine_phosphokinase`, `ejection_fraction`, `platelets`, `serum_creatinine`, `serum_sodium`, `time` |
| **Kategorik/Biner** | `anaemia`, `diabetes`, `high_blood_pressure`, `smoking` |
| **Target** | `DEATH_EVENT` (0 = pasien bertahan hidup, 1 = pasien meninggal) |

---

## 🚀 Tahap 1 — Eksplorasi dengan AutoML

```python
from jcopml.automl import AutoClassifier

model = AutoClassifier(
    numerical_columns=["age", "creatinine_phosphokinase", "ejection_fraction",
                        "platelets", "serum_creatinine", "serum_sodium", "time"],
    categorical_columns=["anaemia", "diabetes", "high_blood_pressure", "smoking"]
)

model.fit(X, y, cv=5)
```

`AutoClassifier` secara otomatis akan:
1. Melakukan preprocessing yang sesuai untuk masing-masing tipe kolom (numerik vs kategorik)
2. Melatih **beberapa algoritma klasifikasi** sekaligus (misalnya Logistic Regression, Random Forest, XGBoost, dll.)
3. Melakukan hyperparameter tuning otomatis dengan **5-fold cross-validation**
4. Membandingkan performa antar algoritma dalam satu proses `fit()`

### Melihat Hasil Perbandingan Algoritma
```python
model.plot_results()
```
Menampilkan visualisasi perbandingan skor antar algoritma yang dicoba — memudahkan memilih kandidat model terbaik secara objektif.

### Feature Importance
```python
model.mean_score_decrease()
```
Menampilkan fitur-fitur medis yang paling berpengaruh terhadap prediksi risiko kematian, berguna untuk interpretasi klinis (*explainability*) — penting dalam domain medis di mana keputusan model harus bisa dijelaskan.

---

##  Tahap 2 — Fokus & Perkuat Tuning pada Model Terbaik

Setelah tahap eksplorasi menunjukkan **XGBoost** sebagai algoritma dengan performa terbaik, tuning dilanjutkan dengan berfokus hanya pada algoritma tersebut, dengan jumlah percobaan (*trial*) yang jauh lebih banyak:

```python
model.fit(X, y, cv=5, algo=["xgb"], n_trial=100)
```

| Parameter | Detail |
|---|---|
| **`algo=["xgb"]`** | Membatasi tuning hanya pada XGBoost (algoritma terbaik dari tahap eksplorasi) |
| **`n_trial=100`** | 100 kali percobaan kombinasi hyperparameter — jauh lebih ekstensif dibanding tahap eksplorasi awal |
| **`cv=5`** | 5-fold cross-validation untuk memastikan hasil general dan stabil |

>  **Insight kunci:** Strategi *"explore broadly, then focus deeply"* ini efisien untuk workflow machine learning — tahap pertama AutoML membantu **menyaring** kandidat algoritma terbaik dengan cepat, lalu tahap kedua **mengoptimalkan** algoritma terpilih tersebut secara maksimal tanpa membuang waktu komputasi pada algoritma yang kurang menjanjikan.

---

##  Tech Stack

- **jcopml (`AutoClassifier`)** — Otomasi preprocessing, training, dan tuning multi-algoritma
- **XGBoost** — Algoritma akhir yang dipilih setelah proses AutoML
- **pandas** & **NumPy** — Manipulasi data
- **scikit-learn** — Cross-validation & metrik evaluasi (di balik layar `jcopml`)
- **Matplotlib** — Visualisasi hasil perbandingan model

---

##  Lisensi

Proyek ini menggunakan lisensi **MIT** — bebas digunakan, dimodifikasi, dan didistribusikan ulang untuk keperluan pembelajaran maupun pengembangan lebih lanjut.

---

##  Kontribusi

Kontribusi, saran, dan diskusi sangat terbuka! Silakan buat *issue* atau *pull request* — misalnya menambahkan evaluasi lanjutan (confusion matrix, ROC curve) untuk model XGBoost final, atau membandingkan hasil AutoML dengan model yang dituning manual.

---
