from joblib import load
import seaborn as sns
from sklearn.metrics import confusion_matrix, mean_absolute_error
import librosa
from matplotlib import pyplot as plt
import numpy as np

model = load('car_count_model.joblib')

# List of test audio files
test_audio_paths = ['test_sounds/1 car_t.WAV', 'test_sounds/1 car_noise.WAV',
                    'test_sounds/2 car_t.WAV', 'test_sounds/2 car_noise.WAV', 
                    'test_sounds/3 car_t.WAV', 'test_sounds/3 car_noise.WAV', 
                    'test_sounds/4 car_t.WAV', 'test_sounds/4 car_noise.WAV', 
                    'test_sounds/5 car_t.WAV', 'test_sounds/5 car_noise.WAV']
test_labels = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]  # Actual number of cars for evaluation, if available

predictions = []
# Process each test audio file
for audio_path in test_audio_paths:
    # Load the audio file
    y_test, sr_test = librosa.load(audio_path, sr=None)
    # Extract MFCCs
    mfcc_test = librosa.feature.mfcc(y=y_test, sr=sr_test, n_mfcc=13)
    # Get the mean of the coefficients to use as a single feature vector
    mfcc_scaled_test = np.mean(mfcc_test.T, axis=0)
    # Reshape the features to match the expected input for the model (1 sample, n features)
    mfcc_scaled_test = mfcc_scaled_test.reshape(1, -1)
    # Predict the label for the test audio file
    prediction = model.predict(mfcc_scaled_test)
    predictions.append(prediction[0])

# Output predictions for the test dataset
for pred, actual in zip(predictions, test_labels):
    print(f"Predicted: {pred}, Actual: {actual}")

# Calculate the Mean Absolute Error (MAE) for the test dataset
mae_test = mean_absolute_error(test_labels, predictions)
print(f"Mean Absolute Error (MAE) on new test set: {mae_test:.2f}")

# Generate and display a confusion matrix for the new test data
test_conf_matrix = confusion_matrix(test_labels, predictions)
plt.figure(figsize=(10, 7))
sns.heatmap(test_conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=[1, 2, 3, 4, 5], yticklabels=[1, 2, 3, 4, 5])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix for Test Audio Paths')
plt.show()