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
* **Custom Dataset Upload:** Upload your own CSV files. The platform automatically detects numeric/categorical columns, handles missing values, applies scaling, lets you select target variables, and creates decision boundary/confusion matrix visualizations dynamically.

---

## 📁 Project Structure

```text
AI_Lab_Studio/
├── app.py                      # Main Streamlit application entry point
├── run.sh                      # One-click startup script for Linux/macOS
├── requirements.txt            # Python dependencies
├── .streamlit/
│   └── config.toml             # Custom theme configuration (Dark Mode UI)
└── modules/
    ├── __init__.py
    ├── intelligent_agents.py   # Code for agent simulations
    ├── search_algorithms.py    # Code for pathfinding/search visualization
    ├── optimization.py         # Code for SA, Hill Climbing, N-Queens
    ├── genetic_algorithm.py    # Code for genetic evolution simulation
    ├── csp.py                  # Code for Map Coloring, N-Queens, Sudoku
    └── ml_playground.py        # Code for Scikit-learn playground & CSV Uploads
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
You can use the provided bash script to automatically activate the environment and launch the app in your browser:
```bash
chmod +x run.sh
./run.sh
```

### 3. Manual Launch (Windows / All Platforms)
If you prefer to run it manually:

**Create and activate a virtual environment:**
```bash
# Create virtual environment
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

---

## 🛠️ Technologies Used
* **[Python 3](https://www.python.org/)** - Core programming language.
* **[Streamlit](https://streamlit.io/)** - Front-end web framework for the interactive UI.
* **[Plotly](https://plotly.com/python/)** - Beautiful, interactive charting and graphing.
* **[NetworkX](https://networkx.org/)** - For generating and traversing search algorithm graphs.
* **[Scikit-learn](https://scikit-learn.org/)** - Machine learning model implementations.
* **NumPy & Pandas** - Advanced data manipulation and numerical operations.
