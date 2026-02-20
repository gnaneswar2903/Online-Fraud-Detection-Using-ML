from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# ==============================
# Load Model & Features
# ==============================
print("Loading model...")

model = joblib.load("model/fraud_detection_model_optimized.pkl")
features = joblib.load("model/model_features.pkl")

print("Model loaded successfully!")

# ==============================
# HOME ROUTE
# ==============================
@app.route("/")
def home():
    return render_template("index.html")


# ==============================
# PREDICTION ROUTE
# ==============================
@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":
        try:
            # --------------------------
            # Get form values
            # --------------------------
            step = float(request.form["step"])
            amount = float(request.form["amount"])
            oldbalanceOrg = float(request.form["oldbalanceOrg"])
            newbalanceOrig = float(request.form["newbalanceOrig"])
            oldbalanceDest = float(request.form["oldbalanceDest"])
            newbalanceDest = float(request.form["newbalanceDest"])
            transaction_type = request.form["type"]

            # --------------------------
            # Create input dictionary
            # --------------------------
            input_dict = {
                "step": step,
                "amount": amount,
                "oldbalanceOrg": oldbalanceOrg,
                "newbalanceOrig": newbalanceOrig,
                "oldbalanceDest": oldbalanceDest,
                "newbalanceDest": newbalanceDest
            }

            # --------------------------
            # Handle One-Hot Encoding
            # --------------------------
            for col in features:
                if "type_" in col:
                    input_dict[col] = 1 if col == f"type_{transaction_type}" else 0

            # Fill missing columns
            for col in features:
                if col not in input_dict:
                    input_dict[col] = 0

            # Convert to DataFrame
            input_df = pd.DataFrame([input_dict])
            input_df = input_df[features]

            # --------------------------
            # Model Prediction
            # --------------------------
            probability = model.predict_proba(input_df)[0][1]

            # --------------------------
            # Rule-Based Fraud Boost
            # --------------------------
            rule_fraud = False

            if (
                transaction_type in ["TRANSFER", "CASH_OUT"] and
                oldbalanceOrg > 0 and
                amount > 100000 and
                (oldbalanceOrg - amount != newbalanceOrig) and
                (newbalanceDest - oldbalanceDest != amount)
            ):
                rule_fraud = True

            # --------------------------
            # Final Decision
            # --------------------------
            if probability > 0.15 or rule_fraud:
                result = "Fraud"
            else:
                result = "Safe"

            # --------------------------
            # Send to result page
            # --------------------------
            return render_template(
                "result.html",
                result=result,
                probability=round(probability * 100, 2)
            )

        except Exception as e:
            return f"Error occurred: {str(e)}"

    return render_template("predict.html")


# ==============================
# RUN SERVER
# ==============================
if __name__ == "__main__":
    app.run(debug=True)
