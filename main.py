from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib

app = FastAPI(title="FraudShield AI Backend")

# ----------------------------
# CORS
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Load ML model
# ----------------------------
model = joblib.load("model.pkl")

# ----------------------------
# Simulated transaction database (FAKE DB)
# ----------------------------
TRANSACTIONS_DB = [
    {
        "account_id": "ACC1001",
        "amount": 500,
        "country": "India",
        "transactions_last_24h": 1,
        "is_new_device": False
    },
    {
        "account_id": "ACC1002",
        "amount": 45000,
        "country": "India",
        "transactions_last_24h": 4,
        "is_new_device": False
    },
    {
        "account_id": "ACC1003",
        "amount": 90000,
        "country": "Nigeria",
        "transactions_last_24h": 8,
        "is_new_device": True
    }
]

# ----------------------------
# Data model
# ----------------------------
class Transaction(BaseModel):
    account_id: str
    amount: float
    country: str
    transactions_last_24h: int
    is_new_device: bool

# ----------------------------
# Root API
# ----------------------------
@app.get("/")
def read_root():
    return {"message": "FraudShield AI Backend is running with ML"}

# ----------------------------
# Fraud check API (ML + Rules)
# ----------------------------
@app.post("/check-transaction")
def check_transaction(txn: Transaction):

    # ML prediction
    features = [[
        txn.amount,
        txn.transactions_last_24h,
        int(txn.is_new_device)
    ]]

    ml_prediction = model.predict(features)[0]
    ml_score = 70 if ml_prediction == 1 else 20

    # Rule-based score
    rule_score = 0
    if txn.amount > 50000:
        rule_score += 30
    if txn.country.lower() in ["nigeria", "pakistan"]:
        rule_score += 25
    if txn.transactions_last_24h > 5:
        rule_score += 20
    if txn.is_new_device:
        rule_score += 15

    # Final score
    final_score = min(ml_score + rule_score, 100)

    if final_score >= 70:
        level = "HIGH"
    elif final_score >= 30:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {
        "risk_score": final_score,
        "risk_level": level,
        "ml_prediction": "FRAUD" if ml_prediction == 1 else "NORMAL"
    }

# ----------------------------
# Auto-check API (no manual input)
# ----------------------------
@app.get("/auto-check")
def auto_check_transactions():
    results = []

    for txn in TRANSACTIONS_DB:
        features = [[
            txn["amount"],
            txn["transactions_last_24h"],
            int(txn["is_new_device"])
        ]]

        ml_prediction = model.predict(features)[0]
        ml_score = 70 if ml_prediction == 1 else 20

        rule_score = 0
        if txn["amount"] > 50000:
            rule_score += 30
        if txn["country"].lower() in ["nigeria", "pakistan"]:
            rule_score += 25
        if txn["transactions_last_24h"] > 5:
            rule_score += 20
        if txn["is_new_device"]:
            rule_score += 15

        final_score = min(ml_score + rule_score, 100)
        level = "HIGH" if final_score >= 70 else "MEDIUM" if final_score >= 30 else "LOW"

        results.append({
            "account_id": txn["account_id"],
            "amount": txn["amount"],
            "risk_score": final_score,
            "risk_level": level,
            "ml_prediction": "FRAUD" if ml_prediction == 1 else "NORMAL"
        })

    return results
