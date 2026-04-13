import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

# Sample dataset
data = {
    "hour": [8, 9, 10, 17, 18, 19],
    "day": [1, 1, 1, 5, 5, 5],
    "vehicles": [200, 250, 180, 300, 350, 330]
}

df = pd.DataFrame(data)

X = df[["hour", "day"]]
y = df["vehicles"]

model = RandomForestRegressor()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved!")