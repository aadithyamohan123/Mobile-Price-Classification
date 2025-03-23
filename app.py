from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load models and scalers
logmodel = joblib.load("logmodel.pkl")
nbmodel = joblib.load("nbmodel.pkl")
knnmodel = joblib.load("knn.pkl")
dtmodel = joblib.load("decisiontree.pkl")
standard_scaler = joblib.load("standard_scaler.pkl")
minmax_scaler = joblib.load("minmax_scaler.pkl")

# Feature ranges for sliders
feature_ranges = {
    "battery_power": {"min": 501, "max": 1998, "mean": 1238.52},
    "int_memory": {"min": 2, "max": 64, "mean": 32.05},
    "px_width": {"min": 500, "max": 1998, "mean": 1251.52},
    "px_height": {"min": 0, "max": 1960, "mean": 645.11},
    "n_cores": {"min": 1, "max": 8, "mean": 4.52},
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/logistic_regression', methods=['GET', 'POST'])
def logistic_regression():
    if request.method == 'POST':
        features = get_features_for_19(request.form)  # Use 19 features
        scaled_features = standard_scaler.transform([features])  # Use StandardScaler
        prediction = logmodel.predict(scaled_features)[0]
        return render_template('logistic_regression.html', prediction=prediction, feature_ranges=feature_ranges)
    return render_template('logistic_regression.html', prediction=None, feature_ranges=feature_ranges)

@app.route('/naive_bayes', methods=['GET', 'POST'])
def naive_bayes():
    if request.method == 'POST':
        features = get_features_for_19(request.form)  # Use 19 features
        scaled_features = standard_scaler.transform([features])  # Use StandardScaler
        prediction = nbmodel.predict(scaled_features)[0]
        return render_template('naive_bayes.html', prediction=prediction, feature_ranges=feature_ranges)
    return render_template('naive_bayes.html', prediction=None, feature_ranges=feature_ranges)

@app.route('/knn', methods=['GET', 'POST'])
def knn():
    if request.method == 'POST':
        features = get_features_for_19(request.form)  # Use 19 features
        scaled_features = standard_scaler.transform([features])  # Use StandardScaler
        prediction = knnmodel.predict(scaled_features)[0]
        return render_template('knn.html', prediction=prediction, feature_ranges=feature_ranges)
    return render_template('knn.html', prediction=None, feature_ranges=feature_ranges)

@app.route('/decision_tree', methods=['GET', 'POST'])
def decision_tree():
    if request.method == 'POST':
        features = get_features_for_22(request.form)  # Use 22 features
        scaled_features = minmax_scaler.transform([features])  # Use MinMaxScaler
        prediction = dtmodel.predict(scaled_features)[0]
        return render_template('decision_tree.html', prediction=prediction, feature_ranges=feature_ranges)
    return render_template('decision_tree.html', prediction=None, feature_ranges=feature_ranges)

def get_features_for_19(form):
    # Extract and format 19 features
    features = [
        int(form.get('battery_power', 1238)),  # battery_power
        int(form.get('blue', 0)),              # blue
        1.52,                                 # clock_speed (default mean)
        int(form.get('dual_sim', 0)),         # dual_sim
        4.31,                                # fc (default mean)
        int(form.get('four_g', 0)),           # four_g
        int(form.get('int_memory', 32)),      # int_memory
        140.25,                              # mobile_wt (default mean)
        int(form.get('n_cores', 4)),          # n_cores
        9.92,                                # pc (default mean)
        int(form.get('px_height', 645)),      # px_height
        int(form.get('px_width', 1251)),     # px_width
        2124.21,                             # ram (default mean)
        12.31,                               # sc_h (default mean)
        5.77,                                # sc_w (default mean)
        11.01,                               # talk_time (default mean)
        int(form.get('three_g', 0)),         # three_g
        0.50,                                # touch_screen (default mean)
        int(form.get('wifi', 0)),            # wifi
    ]
    return features

def get_features_for_22(form):
    # Extract and format 19 features first
    features_19 = get_features_for_19(form)

    # Calculate additional features for Decision Tree
    px_area = int(form.get('px_height', 645)) * int(form.get('px_width', 1251))  # px_area
    battery_per_core = int(form.get('battery_power', 1238)) / max(1, int(form.get('n_cores', 4)))  # battery_per_core
    memory_density = 2124.21 / max(1, int(form.get('int_memory', 32)))  # memory_density

    # Combine all 22 features
    features_22 = features_19 + [px_area, battery_per_core, memory_density]
    return features_22

if __name__ == '__main__':
    app.run(debug=True)