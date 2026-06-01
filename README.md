# 🧠 AI Lab Studio

**AI Lab Studio** is an interactive educational platform developed using Python to visualize, simulate, and experiment with core Artificial Intelligence algorithms and Machine Learning techniques. The project transforms traditional console-based laboratory tasks into a modern graphical learning environment with rich UI/UX features.

---

## ✨ Key Features

### 1. 🤖 Intelligent Agent Simulation
* Simulate Vacuum Cleaner agents, Reflex agents, and Goal-based agents.
* Interactive PEAS (Performance, Environment, Actuators, Sensors) representation visualization.

### 2. 🔍 Search Algorithm Visualizer
* Step-by-step interactive graph traversal with animation controls.
* Includes Breadth-First Search (BFS), Depth-First Search (DFS), Depth-Limited Search (DLS), Iterative Deepening Search (IDS), and A* Search.

### 3. ⚙️ Optimization Algorithms
* Visual and interactive solvers for Hill Climbing and Simulated Annealing.
* Combinatorial optimization solving the classic 8-Queens Problem.

### 4. 🧬 Genetic Algorithm Simulator
* Interactive chromosome visualization.
* Watch populations evolve over generations through Selection, Crossover, and Mutation.
* Interactive crossover and mutation sandbox.

### 5. 🧩 Constraint Satisfaction Problems (CSP)
* Solvers using Backtracking and Constraint Propagation.
* Visualizations for Australia Map Coloring, N-Queens, and Sudoku.

### 6. 🤖 Machine Learning Playground
* Train, inspect, and compare classic ML models: K-Nearest Neighbors (KNN), Naïve Bayes, K-Means Clustering, and Logistic Regression.
* **Dataset Upload System:** Upload your own CSV, TXT, or XLSX files directly inside any ML tab. Automatically previews, cleans, and allows feature/target selection before training.
* Built-in datasets (Iris, Wine, Breast Cancer, Digits) remain available alongside the upload option.

### 7. 🔐 Authentication System
* Secure login and signup with username/email + password.
* Guest mode for exploration without an account.
* Session management via Streamlit session_state.

### 8. 🗂️ Student Workspace
* Personal area to save, manage, and reload experiments, datasets, and trained models.
* Experiment comparison charts (accuracy, cost, path length) across multiple runs.
* Save trained KNN, Naive Bayes, K-Means, and Logistic Regression models using joblib.

---

## 📦 Dataset Upload System

### Supported File Formats
| Format | Extension | Notes |
|--------|-----------|-------|
| CSV | `.csv` | Comma-separated values |
| Text | `.txt` | Comma-separated plain text |
| Excel | `.xlsx` | Microsoft Excel workbook |

### Upload Workflow
1. Navigate to **ML Playground** → choose any model tab (KNN, Naïve Bayes, K-Means, Logistic Regression).
2. At the top of the tab, select **"Upload Your Own Dataset"** from the Dataset Source radio.
3. Click the file uploader and select your file.
4. The app automatically displays:
   - Dataset name, row count, column count, file type
   - First 10 rows preview table
   - Column statistics (dtype, non-null count, missing values)

### Feature & Target Selection
* **Feature columns** — choose one or more numeric columns as model inputs.
* **Target column** — choose the output/label column (for classifiers); not required for K-Means.
* Non-numeric targets are automatically label-encoded.

### Data Preprocessing Options
| Option | Description |
|--------|-------------|
| Keep As Is | No changes to missing values |
| Remove Rows With Missing Values | Drops any row containing NaN |
| Fill Numeric With Mean | Replaces NaN with column mean |
| Fill Numeric With Median | Replaces NaN with column median |
| None (scaling) | Raw feature values used |
| StandardScaler | Zero mean, unit variance |
| MinMaxScaler | Scale features to [0, 1] range |

### ML Compatibility
All four ML models fully support uploaded datasets:

| Model | Accuracy | Confusion Matrix | Notes |
|-------|----------|-----------------|-------|
| KNN | ✅ | ✅ | Configurable K |
| Naïve Bayes | ✅ | ✅ | Classification report |
| K-Means | — | — | Elbow curve shown |
| Logistic Regression | ✅ | ✅ | Configurable C & solver |

### Saving Uploaded Datasets to Workspace
After uploading and configuring your dataset, expand **"💾 Save Dataset to Workspace"** at the bottom of the dataset panel. Logged-in users can save the dataset name, upload date, row/column counts, and feature list. Saved datasets appear in **Student Workspace → Datasets** and can be reloaded later.

### Error Handling
| Situation | Behaviour |
|-----------|-----------|
| Invalid file type | Error message shown |
| Empty dataset | Warning shown |
| No target selected (classifier) | Warning shown |
| Non-numeric target | Auto label-encoding applied |
| File > 20 MB | Size warning shown |
| Too few rows after cleaning | Error with guidance |

---

## 📁 Project Structure

```text
AI_Lab_Studio/
├── app.py                        # Main Streamlit application entry point
├── run.sh                        # One-click startup script for Linux/macOS
├── requirements.txt              # Python dependencies
├── ai_lab_studio.db              # SQLite database (auto-created on first run)
├── .streamlit/
│   └── config.toml               # Custom theme configuration (Dark Mode UI)
├── auth/
│   ├── __init__.py
│   └── auth_ui.py                # Login, Signup, Logout, Guest Mode
├── storage/
│   ├── __init__.py
│   ├── db.py                     # SQLite schema bootstrap
│   ├── users.py                  # User CRUD + password hashing
│   ├── experiments.py            # Experiment save/list/delete
│   ├── datasets.py               # Dataset metadata + file management
│   └── models.py                 # Model metadata + joblib persistence
├── workspace/
│   ├── __init__.py
│   ├── workspace_ui.py           # Student Workspace page (4 tabs)
│   ├── dataset_upload.py         # Generic reusable dataset upload widget
│   ├── ml_dataset.py             # ML-playground-specific dataset loader
│   └── save_helpers.py           # Save Experiment / Save Model buttons
├── user_datasets/                # Uploaded dataset files (auto-created)
├── user_models/                  # Saved model files (auto-created)
└── modules/
    ├── __init__.py
    ├── intelligent_agents.py     # Agent simulations
    ├── search_algorithms.py      # BFS, DFS, DLS, IDS, A* visualizer
    ├── optimization.py           # Hill Climbing, SA, 8-Queens
    ├── genetic_algorithm.py      # Genetic evolution simulation
    ├── csp.py                    # Map Coloring, N-Queens, Sudoku
    └── ml_playground.py          # ML models + dataset upload
```

---

## 🚀 Quick Start Guide

### Prerequisites
* Python 3.8 or newer.

### 1. Navigate to the App Repository
```bash
cd /path/to/AI_Lab_Studio
```

### 2. Easy Launch (Linux/macOS)
```bash
chmod +x run.sh
./run.sh
```

### 3. Manual Launch (Windows / All Platforms)

**Create and activate a virtual environment:**
```bash
python -m venv venv

# Activate on Linux/macOS:
source venv/bin/activate

# Activate on Windows:
# venv\Scripts\activate
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the Streamlit application:**
```bash
streamlit run app.py
```

The application will spin up and become accessible at `http://localhost:8501`.

### 4. First-Time Setup
On first launch the app will:
1. Show a **Login / Signup** screen — create an account or continue as guest.
2. Auto-create the SQLite database (`ai_lab_studio.db`) with all required tables.
3. Create `user_datasets/` and `user_models/` directories for file storage.

---

## 📖 Usage Instructions

### Using the ML Playground with Your Own Dataset
1. Log in (or continue as guest to explore).
2. Go to **🤖 ML Playground** in the sidebar.
3. Choose a model tab (e.g., **🔵 KNN**).
4. At the top of the tab, select **"Upload Your Own Dataset"**.
5. Upload a CSV/TXT/XLSX file.
6. Review the preview and configure:
   - Missing value handling
   - Feature scaling
   - Feature columns (inputs)
   - Target column (output label)
7. The model trains automatically and displays accuracy + confusion matrix.
8. Optionally save the experiment or trained model to your **Student Workspace**.

### Saving & Comparing Experiments
* After any algorithm run, expand **"💾 Save this experiment"** to persist results.
* Open **Student Workspace** (sidebar button) → **📊 Compare** tab.
* Select multiple algorithms to compare accuracy, cost, and path length side-by-side.

---

## 🛠️ Technologies Used
* **[Python 3](https://www.python.org/)** - Core programming language.
* **[Streamlit](https://streamlit.io/)** - Front-end web framework for the interactive UI.
* **[Plotly](https://plotly.com/python/)** - Beautiful, interactive charting and graphing.
* **[NetworkX](https://networkx.org/)** - For generating and traversing search algorithm graphs.
* **[Scikit-learn](https://scikit-learn.org/)** - Machine learning model implementations.
* **[NumPy & Pandas](https://numpy.org/)** - Advanced data manipulation and numerical operations.
* **[SQLite3](https://docs.python.org/3/library/sqlite3.html)** - Lightweight database for user data, experiments, datasets, and models.
* **[joblib](https://joblib.readthedocs.io/)** - Efficient model serialisation and persistence.

---

## 📸 Screenshots

| Screen | Description |
|--------|-------------|
| Login / Signup | Auth gate before accessing the app |
| Home Dashboard | Platform overview with architecture diagram |
| Search Algorithms | Animated BFS/DFS/A* graph traversal |
| ML Playground | Decision boundary plots + dataset upload |
| Student Workspace | Saved experiments, datasets, models & comparison |
