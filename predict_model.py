import pandas as pd
import numpy as np
from keras.models import load_model
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from train_model import preprocess_text, tokenize_and_pad
import os
from nltk.corpus import stopwords
import nltk
from keras.preprocessing.text import Tokenizer
from train_model import preprocess_text, tokenize_and_pad

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords', quiet=True)

# Load the trained model
model = load_model('models/hate_speech_model.h5')

# Load the tokenizer
with open('models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Ensure the loaded object is a Tokenizer
if not isinstance(tokenizer, Tokenizer):
    raise TypeError("The loaded object is not a Tokenizer instance")

# Load and preprocess the test data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, sep=',')
    df['label'] = df['label'].map({'hate': 1, 'noHate': 0})
    df['text'] = df['content'].apply(preprocess_text)
    return df['text'].values, df['label'].values, df['content'].values

# Load test data
X_test, y_test, original_texts = load_and_preprocess_data('data/modified_dataset/hate_speech_test_dataset.csv')

# Tokenize and pad the test data
X_test_padded, _ = tokenize_and_pad(X_test, tokenizer=tokenizer)

# Make predictions
y_pred = model.predict(X_test_padded)
y_pred_classes = (y_pred > 0.5).astype(int).flatten()

# Generate classification report
report = classification_report(y_test, y_pred_classes)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)

# Create a DataFrame with original texts, true labels, and predicted labels
results_df = pd.DataFrame({
    'Text': original_texts,
    'True Label': ['hate' if label == 1 else 'noHate' for label in y_test],
    'Predicted Label': ['hate' if label == 1 else 'noHate' for label in y_pred_classes],
    'Prediction Probability': y_pred.flatten()
})

# Save results to files
with open('results/classification_report.txt', 'w') as f:
    f.write(report)

# Plot and save confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('results/confusion_matrix.png')
plt.close()

# Save results DataFrame to CSV
results_df.to_csv('results/prediction_results.csv', index=False)

# Generate a formatted text report
with open('results/full_report.txt', 'w') as f:
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(cm))
    f.write("\n\nSample Predictions:\n")
    for _, row in results_df.head(10).iterrows():
        f.write(f"Text: {row['Text'][:100]}...\n")
        f.write(f"True Label: {row['True Label']}\n")
        f.write(f"Predicted Label: {row['Predicted Label']}\n")
        f.write(f"Prediction Probability: {row['Prediction Probability']:.4f}\n")
        f.write("---\n")

print("Results have been saved in the 'results' folder.")
