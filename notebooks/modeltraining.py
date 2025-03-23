import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import logging
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load data
df = pd.read_csv('/workspaces/Mobile-Price-Classification/data/train.csv')
df = df.drop(columns=['touch_screen', 'mobile_wt', 'clock_speed'])

# Feature engineering
df['px_area'] = df['px_height'] * df['px_width']
df['battery_per_core'] = df['battery_power'] / df['n_cores']
df['memory_density'] = df['ram'] / df['int_memory']
df['ram'] = np.log(df['ram']) / np.log(2)

# Separate features and target
y = df['price_range']
x = df.drop(columns=['price_range'])

# Feature scaling
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.33, random_state=42)

# Logistic Regression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
y_pred = logmodel.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
'''with open('logmodel.pkl', 'wb') as model_file:
    pickle.dump(logmodel, model_file)'''

# Naive Bayes
nbmodel = GaussianNB()
nbmodel.fit(X_train, y_train)
y_pred = nbmodel.predict(X_test)
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred))
'''with open('nbmodel.pkl', 'wb') as model_file:
    pickle.dump(nbmodel, model_file)'''

# KNN with Cross-Validation
k_values = range(1, 50)  # Reduced range for faster computation
cv_scores = [cross_val_score(KNeighborsClassifier(n_neighbors=k), x_scaled, y, cv=5).mean() for k in k_values]
optimal_k = k_values[np.argmax(cv_scores)]
print(f"Optimal K: {optimal_k}")

knnmodel = KNeighborsClassifier(n_neighbors=optimal_k)
knnmodel.fit(X_train, y_train)
y_pred = knnmodel.predict(X_test)
print("KNN Accuracy:", accuracy_score(y_test, y_pred))
'''with open('knn.pkl', 'wb') as model_file:
    pickle.dump(knnmodel, model_file)'''

# Decision Tree with Hyperparameter Tuning
param_grid = {
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']  # Removed 'auto'
}
grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(f"Best Parameters for Decision Tree: {grid_search.best_params_}")

dtmodel = grid_search.best_estimator_
y_pred = dtmodel.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))
'''with open('decisiontree.pkl', 'wb') as model_file:
    pickle.dump(dtmodel, model_file)'''