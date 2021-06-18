from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd

def predict(model, test_data):
    predict_result = model.predict(test_data)
    anomaly_scores = model.score_samples(test_data) * -1
    return predict_result, anomaly_scores


def train(train_data):
    model = IsolationForest(random_state=0)
    model = model.fit(train_data)
    return model


