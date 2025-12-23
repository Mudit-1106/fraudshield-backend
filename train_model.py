import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# ----------------------------
# Step 1: Create sample training data
# ----------------------------
data = {
    "amount": [500, 2000, 60000, 75000, 1200, 90000, 300, 40000],
    "transactions_last_24h": [1, 2, 6, 8, 1, 10, 0, 5],
    "is_new_device": [0, 0, 1, 1, 0, 1, 0, 1],
    "is_fraud": [0, 0, 1, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# ----------------------------
# Step 2: Split features & label
# ----------------------------
X = df[["amount", "transactions_last_24h", "is_new_device"]]
y = df["is_fraud"]

# ----------------------------
# Step 3: Train model
# ----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# ----------------------------
# Step 4: Save model
# ----------------------------
joblib.dump(model, "model.pkl")

print("✅ ML model trained and saved as model.pkl")
