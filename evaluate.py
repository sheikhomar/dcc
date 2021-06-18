import tensorflow as tf
from sklearn.metrics import accuracy_score
import numpy as np
import json
import sys

user_predictions = str(sys.argv[1])
test_data = str(sys.argv[2] + '/label_book')
tf.random.set_seed(123)

with open(user_predictions) as f:
    y_pred = json.load(f)

test = tf.keras.preprocessing.image_dataset_from_directory(
    test_data,
    labels="inferred",
    label_mode="categorical",
    class_names=["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"],
    shuffle=False,
    seed=123,
    batch_size=8,
    image_size=(32, 32),
)

y = []
for element in test.unbatch():
    y.append(element[1])

y_true = np.array(y).argmax(axis=1)

score = {'accuracy': accuracy_score(y_true, y_pred)}

with open('result.json', 'w') as outfile:
    json.dump(score, outfile)
