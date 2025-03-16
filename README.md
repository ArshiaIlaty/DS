# Credit Card Fraud Detection Solution

## 📌 Overview
This solution helps detect fraudulent credit card transactions using machine learning techniques. Please follow the steps below to set up your environment, install the required packages, and understand the project structure.

---

## 🛠️ Setup and Requirements

### 🔹 Create a Virtual Environment (Recommended)
To avoid conflicts with other dependencies, it's highly recommended to create a new virtual environment before installing the required packages.

#### **Using Conda (Recommended)**
```bash
# Create and activate a new environment
conda create -n fraud_detection python=3.10 -y
conda activate fraud_detection
```

#### **Using venv**
```bash
# Create a virtual environment
python -m venv fraud_env

# Activate the environment
# On Windows:
fraud_env\\Scripts\\activate
# On Mac/Linux:
source fraud_env/bin/activate
```

---

### 📦 Install Required Packages
After setting up the environment, install the necessary dependencies.

#### **Install from `requirements.txt` (Preferred)**
```bash
pip install -r requirements.txt
```
#### **Manually Install Core Dependencies**
Alternatively, you can install the required packages individually:
```bash
# Core libraries
pip install pandas numpy matplotlib seaborn scikit-learn tqdm ipykernel jupyter

# Handling class imbalance
pip install imbalanced-learn

# XGBoost for advanced modeling
pip install xgboost
```

---

## 📂 File Organization
Ensure all provided files are placed in the same directory for seamless execution.

| File Name           | Description |
|---------------------|-------------|
| `data_loader.py`    | Loads and describes the dataset |
| `visualization.py`  | Generates visualizations and insights |
| `data_wrangling.py` | Handles duplicates and recurring transactions |
| `modeling.py`       | Builds, trains, and evaluates machine learning models |
| `main.py`           | Main execution script that orchestrates the workflow |

---

## Next Steps
1. Load and explore your dataset using `data_loader.py`.
2. Visualize the data with `visualization.py` to gain insights.
3. Preprocess and clean your dataset using `data_wrangling.py`.
4. Train and evaluate models with `modeling.py`.

---
## Running the Pipeline
### Command Line Usage

```bash
# Run the complete pipeline
python main.py --data transactions.txt --output-dir results

# To skip plots and use cached results
python main.py --data transactions.txt --skip-plots --output-dir results

# To run the code on a subsample of the dataset
python main.py --data transactions.txt --sample-size 100000
```

### Module-specific Commands

```bash
# Analyze duplicate transactions
python data_wrangling.py --data transactions.txt

# Generate visualizations only
python visualization.py --data transactions.pkl

# Run model training with specific sampling
python modeling.py --data transactions.pkl --sampling smote
```

---
If you wish to convert the notebook to HTML format, you can run the following command:

```bash
jupyter nbconvert file-name.ipynb --to html
```
---

## Need more information?
For any issues or clarifications, feel free to email me.