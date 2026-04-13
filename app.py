from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# ✅ If model not found → create it
if not os.path.exists("model.pkl"):
    data = {
        "hour": [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
        "day":  [1,1,1,1,1,2,2,3,3,4,5,5,5,6,7],
        "vehicles": [100,150,250,300,200,180,160,170,190,210,260,320,350,300,200]
    }

    df = pd.DataFrame(data)
    X = df[["hour", "day"]]
    y = df["vehicles"]

    model = RandomForestRegressor()
    model.fit(X, y)

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

# ✅ Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    hour = int(data["hour"])
    day = int(data["day"])
    city = data["city"]

    prediction = model.predict(np.array([[hour, day]]))[0]

    if city == "Chennai":
        prediction += 80
    elif city == "Coimbatore":
        prediction += 40
    elif city == "Madurai":
        prediction += 30
    elif city == "Salem":
        prediction += 20

    if prediction < 150:
        level = "Low Traffic"
    elif prediction < 300:
        level = "Moderate Traffic"
    else:
        level = "Heavy Traffic"

    return jsonify({
        "vehicles": int(prediction),
        "level": level
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
