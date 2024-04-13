import librosa
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

file_path_true = "C:\\Users\\user\\Documents\\Выявление синтезированного голосаречи с использованием методов машинного обучения (ООО «Даталаб»)\\звук\\true"
file_path_false = "C:\\Users\\user\\Documents\\Выявление синтезированного голосаречи с использованием методов машинного обучения (ООО «Даталаб»)\\звук\\false"

def get_file_names(directory):
    file_names = []
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            file_names.append(filename)
    return file_names

file_names_true = get_file_names(file_path_true)
file_names_false = get_file_names(file_path_false)

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

# Create DataFrame for file names and labels
labels_true = [0] * len(file_names_true)
labels_false = [1] * len(file_names_false)

true_df = pd.DataFrame({'file_name': file_names_true, 'label': labels_true})
false_df = pd.DataFrame({'file_name': file_names_false, 'label': labels_false})

# Merge and shuffle the data
data = pd.concat([true_df, false_df], ignore_index=True)
data = data.sample(frac=1).reset_index(drop=True)  # Shuffle the data

# Save merged and shuffled data to CSV file
data.to_csv('merged_data.csv', index=False)

# Load merged and shuffled data from CSV file
data = pd.read_csv('merged_data.csv')

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data['file_name'], data['label'], test_size=0.2, random_state=42)

# Extract features for train and test sets
X_train = np.array([extract_features(os.path.join(file_path_true, file_name)) for file_name in X_train])
X_test = np.array([extract_features(os.path.join(file_path_true, file_name)) for file_name in X_test])

# Train the classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Save the trained classifier
with open('trained_classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save the training and testing data
with open('X_train.pkl', 'wb') as f:
    pickle.dump(X_train, f)

with open('X_test.pkl', 'wb') as f:
    pickle.dump(X_test, f)

with open('y_train.pkl', 'wb') as f:
    pickle.dump(y_train, f)

with open('y_test.pkl', 'wb') as f:
    pickle.dump(y_test, f)
