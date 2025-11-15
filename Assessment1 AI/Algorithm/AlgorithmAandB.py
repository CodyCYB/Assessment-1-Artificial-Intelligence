import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename


# =============================
# STEP 1: Load & Prepare Dataset
# =============================
def find_file(fname):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, fname),
        os.path.join(script_dir, "data", fname),
        os.path.join(os.getcwd(), fname),
        os.path.join(script_dir, fname.replace("_", " ")),
        os.path.join(script_dir, fname.replace("_", "-")),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    # ask user to select file if not found automatically
    Tk().withdraw()
    selected = askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
    return selected or None


full_path = find_file("Phishing_Email.csv")
if not full_path:
    raise FileNotFoundError("Could not find 'Phishing_Email.csv'. Place it next to the script or select it in the file dialog.")

# read CSV (latin1 fallback)
try:
    df = pd.read_csv(full_path, encoding="latin1")
except Exception as e:
    raise RuntimeError(f"Failed to read CSV at {full_path}: {e}")

df = df.rename(columns={"Email Text": "text", "Email Type": "label"})

# Convert to lowercase and replace NaN
df["text"] = df["text"].astype(str).fillna("")
df["label"] = df["label"].apply(lambda x: 1 if "Phishing" in str(x) else 0)

# Split data into train/test
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.3, random_state=42)


# =====================================
# STEP 2: Algorithm A - Keyword-Based AI
# =====================================
def rule_based_classifier_A(email_text):
    text = str(email_text).lower()
    rules_triggered = 0

    rules = [
        r"(urgent|immediately|verify|warning)",
        r"(free|offer|discount|win)",
        r"http[s]?://|\.com/",
        r"(password|account|login|credit\s?card)",
        r"(horny|xxx|sex|nude)",
        r"[A-Z]{3,}|!{3,}"
    ]

    for rule in rules:
        if re.search(rule, text):
            rules_triggered += 1

    return 1 if rules_triggered >= 2 else 0


# Evaluate Algorithm A
y_pred_A = X_test.apply(rule_based_classifier_A)
results_A = {
    "Accuracy": accuracy_score(y_test, y_pred_A),
    "Precision": precision_score(y_test, y_pred_A, zero_division=0),
    "Recall": recall_score(y_test, y_pred_A, zero_division=0),
    "F1 Score": f1_score(y_test, y_pred_A, zero_division=0)
}


# ============================================
# STEP 3: Algorithm B - Weighted Heuristic AI
# ============================================
def rule_based_classifier_B(email_text):
    text = str(email_text).lower()
    score = 0

    rules = [
        (r"(urgent|immediately|verify|warning)", 2),
        (r"(free|offer|winner|claim|discount)", 1),
        (r"http[s]?://|@[a-z0-9.-]+\.[a-z]{2,}", 3),
        (r"(login|password|reset|account\s?info)", 2),
        (r"(xxx|horny|dating|nude|sex)", 2),
        (r"!{3,}|[A-Z]{5,}", 1),
        (r"\d{3}[-.\s]?\d{3}[-.\s]?\d{4}", 1)
    ]

    for pattern, weight in rules:
        if re.search(pattern, text):
            score += weight

    return 1 if score >= 4 else 0


# Evaluate Algorithm B
y_pred_B = X_test.apply(rule_based_classifier_B)
results_B = {
    "Accuracy": accuracy_score(y_test, y_pred_B),
    "Precision": precision_score(y_test, y_pred_B, zero_division=0),
    "Recall": recall_score(y_test, y_pred_B, zero_division=0),
    "F1 Score": f1_score(y_test, y_pred_B, zero_division=0)
}


# =============================
# STEP 4: Display Final Results
# =============================
print("\nAlgorithm A (Keyword-Based) Results:")
for k, v in results_A.items():
    print(f"  {k}: {v:.3f}")

print("\nAlgorithm B (Weighted Heuristic) Results:")
for k, v in results_B.items():
    print(f"  {k}: {v:.3f}")

# Optional comparison summary
print("\nSummary Comparison:")
print(pd.DataFrame([results_A, results_B], index=["Algorithm A", "Algorithm B"]))
