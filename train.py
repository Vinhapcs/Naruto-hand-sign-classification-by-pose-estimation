import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib


# Path to folder with keypoints (one .npy per label)
KEYPOINTS_FOLDER = 'keypoints_by_label'

X = []  # features
y = []  # labels

# Load each .npy file
for file_name in os.listdir(KEYPOINTS_FOLDER):
    if file_name.endswith('.npy'):
        label = os.path.splitext(file_name)[0]
        file_path = os.path.join(KEYPOINTS_FOLDER, file_name)
        keypoints = np.load(file_path)
        X.extend(keypoints)
        y.extend([label] * len(keypoints))

X = np.array(X)
y = np.array(y)

print(f"Loaded {X.shape[0]} samples across {len(set(y))} classes.")

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Choose your classifier
#model = DecisionTreeClassifier(random_state=42)
model = RandomForestClassifier(n_estimators=200, random_state=42)  # You can try 'linear' kernel too

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

joblib.dump(model, 'hand_sign_model.pkl')
print("Model saved as hand_sign_model.pkl")