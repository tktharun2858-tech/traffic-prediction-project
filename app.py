from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model
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

    # ✅ City-based adjustment (simulated real-world logic)
    if city == "Chennai":
        prediction += 80   # heavy city
    elif city == "Coimbatore":
        prediction += 40
    elif city == "Madurai":
        prediction += 30
    elif city == "Salem":
        prediction += 20

    # Traffic level
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
    app.run(debug=True)