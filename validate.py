from tabnanny import verbose
from tensorflow import keras
import data
import numpy as np
import pickle

# reconstructed_model = keras.models.load_model("classifier")

x = data.Data.x
y = data.Data.y

# print(x[440:441])
# print(y[440:441])


# print(reconstructed_model.predict(x))

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
    print(model.predict(x[458:459]))