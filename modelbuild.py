import librosa
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
    
# Example function to handle non-finite values
def handle_non_finite(y):
    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y)  # Replace NaN with 0 and Inf with large finite numbers
    return y

# Load and Preprocess Audio Files
audio_paths = ['sounds/1 car_1.WAV','sounds/1 car_2.WAV', 'sounds/1 car_3.WAV', 'sounds/1 car_4.WAV', 'sounds/1 car_5.WAV', 
               'sounds/2 car_1.WAV','sounds/2 car_2.WAV', 'sounds/2 car_3.WAV', 'sounds/2 car_4.WAV', 'sounds/2 car_5.WAV', 
               'sounds/3 car_1.WAV','sounds/3 car_2.WAV', 'sounds/3 car_3.WAV', 'sounds/3 car_4.WAV', 'sounds/3 car_5.WAV', 
               'sounds/4 car_1.WAV','sounds/4 car_2.WAV', 'sounds/4 car_3.WAV', 'sounds/4 car_4.WAV', 'sounds/4 car_5.WAV', 
               'sounds/5 car_1.WAV','sounds/5 car_2.WAV', 'sounds/5 car_3.WAV', 'sounds/5 car_4.WAV', 'sounds/5 car_5.WAV', ]  # List of your audio files
labels = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5]  # Corresponding labels indicating the number of cars

features = []
for audio_path, label in zip(audio_paths, labels):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    features.append(mfcc_scaled)

# Prepare Data and Train the Model
# Convert the list of features to a NumPy array
X = np.array(features)
y = np.array(labels)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
dump(model, 'car_count_model.joblib')

# Predict on the test set and evaluate
y_pred = model.predict(X_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Print a classification report
print(classification_report(y_test, y_pred))

# Generate and display a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=[1, 2, 3, 4, 5], yticklabels=[1, 2, 3, 4, 5])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()