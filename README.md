
# 🧠 AutoResearch MLOps Pipeline

An **agentic AutoML + MLOps system** that combines:

* Hyperparameter optimization (Optuna)
* Multi-model training (Neural Network + XGBoost)
* Experiment tracking (MLflow)
* LLM-guided configuration refinement
* Memory-driven learning loop

This project simulates a **self-improving ML system** that learns from past experiments and optimizes itself over time.

---

# 🚀 Features

* 🔁 **AutoML Loop** with Optuna
* 🧠 **LLM-guided tuning** every few trials
* 🗂 **Persistent memory** of past runs
* 📊 **MLflow experiment tracking**
* ⚖️ **Multi-model comparison** (NN vs XGBoost)
* ⏹ **Early stopping logic**
* 🔄 **Continuous improvement pipeline**

---

# 🏗 Project Structure

```
autoresearch_mlops/
│
├── core.py          # Main orchestration (Optuna loop)
├── main.py          # Entry point
├── models.py        # NN + XGBoost training
├── eval.py          # Evaluation (RMSE)
├── tracker.py       # MLflow logging
├── memory.py        # Stores past runs
├── llm.py           # LLM suggestions
├── program.md       # Control logic for LLM
├── mlflow.db        # MLflow database (auto-created)
└── README.md
```

---

# ⚙️ How It Works (Flow)

```
User Data
   ↓
Optuna Trial Loop
   ↓
Config Generation (lr, epochs, model type)
   ↓
LLM Suggestion (every 5 trials)
   ↓
Model Training (NN or XGB)
   ↓
Evaluation (RMSE)
   ↓
MLflow Logging
   ↓
Memory Update
   ↓
Early Stopping Check
   ↓
Repeat
```

---

# 🧠 Key Components

## 1. Optuna (AutoML Engine)

* Explores hyperparameters
* Minimizes RMSE
* Stops early if performance stabilizes

## 2. LLM Suggestion Engine

* Uses past runs (`memory`)
* Reads `program.md`
* Suggests improved configs periodically

## 3. Memory Module

* Stores previous trials
* Enables learning across iterations

## 4. Tracker (MLflow)

* Logs:

  * parameters
  * metrics
  * models
* Enables experiment comparison

---

# 📦 Installation

## 1. Clone repo

```bash
git clone https://github.com/Kamalesh9483/autoresearch-mlops.git
cd autoresearch-mlops
```

---

## 2. Create environment (recommended)

Using `uv`:

```bash
uv venv
uv pip install -r Requirements.txt
```

OR using pip:

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r Requirements.txt
```

---

## 3. Install dependencies

```bash
pip install torch optuna mlflow xgboost pandas scikit-learn
```

---

# ▶️ Running the Project

## Step 1 — Start MLflow Server

⚠️ Use **absolute path** (important)

```powershell
mlflow server --backend-store-uri sqlite:///C:/Autoresearch/autoresearch_mlops/mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```

---

## Step 2 — Run Training

```bash
uv run main.py
```

You should see:

```
Trial 0 finished...
Trial 1 finished...
```

---

## Step 3 — Open MLflow UI

```
http://127.0.0.1:5000
```

Select experiment:

```
autoresearch_mlops
```

---

# 📊 What You’ll See in MLflow

* Experiment: `autoresearch_mlops`
* Runs = Optuna trials
* Metrics:

  * RMSE
* Parameters:

  * model type
  * learning rate
  * epochs
* Model artifacts

---

# ⚠️ Common Issues & Fixes

## ❌ Experiment not showing

✔ Fix: Ensure same tracking URI everywhere

```python
mlflow.set_tracking_uri("sqlite:///C:/Autoresearch/autoresearch_mlops/mlflow.db")
```

---

## ❌ Only "Default" experiment appears

✔ Cause:

* Training and server using different DB

✔ Fix:

* Use **absolute path**

---

## ❌ UnboundLocalError (mlflow)

✔ Cause:

* Import inside function

✔ Fix:

```python
import mlflow.sklearn  # move to top
```

---

## ❌ Model logging warning (predict missing)

✔ Cause:

* PyTorch model used with sklearn logger

✔ Fix (optional):

```python
import mlflow.pytorch
mlflow.pytorch.log_model(model, "model")
```


# 💡 Concept

This project demonstrates:

> **“Self-improving ML systems using feedback loops, memory, and LLM reasoning.”**

It moves beyond static ML pipelines into:

* Adaptive systems
* Intelligent experimentation
* Agentic AI workflows

---


# 📜 License

MIT License


