from flask import Flask, render_template, request
import joblib
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)

# Load model and scaler
model = joblib.load("car_price_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            horsepower = float(request.form["horsepower"])
            curbweight = float(request.form["curbweight"])
            enginesize = float(request.form["enginesize"])

            X = np.array([[horsepower, curbweight, enginesize]])
            X_scaled = scaler.transform(X)
            predicted_price = model.predict(X_scaled)[0]
            prediction = f"${predicted_price:,.2f}"

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
