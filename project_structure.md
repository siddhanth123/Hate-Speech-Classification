# Project Structure and File Descriptions

This document provides an overview of the project's folder structure and key files.

## Folder Structure

1. **Modified Dataset Folder**:

   - Contains two datasets: one for training and one for testing hate speech classification.
   - `dataset_preparation.py`: Script that processes raw data into a pandas-friendly format.

2. **Models Folder**:

   - Contains the saved tokenizer and the pickle file of the best model from training.

3. **Results Folder**:
   - `classification_report.txt`: Detailed metrics on model performance.
   - `confusion_matrix.png`: Visual representation of the model's predictions.
   - `full_report.txt`: Comprehensive report including sample predictions.
   - `prediction_results.csv`: Detailed results of predictions on the test dataset.

## Key Files in Root Directory

- `app.py`: Contains the FastAPI implementation for serving the model.
- `requirements.txt`: Lists all necessary Python packages for the project.
- `train_model.py`: Script for training the hate speech classification model.
- `predict_model.py`: Script for making predictions using the trained model.
- `project.md`: Walkthrough of the approach and thought process.
- `README.md`: Project overview and setup instructions.

## Important Note on Word2Vec Embeddings

For the sake of keeping the project size manageable, the Google News Word2Vec file has been removed from this repository. To run the entire project successfully, you need to download and add this file:

1. Download a pre-trained Word2Vec model:

   - Go to the Google Drive link: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
   - Download the file named "GoogleNews-vectors-negative300.bin.gz"
   - Extract the `.bin` file from the `.gz` archive

2. Place the extracted `.bin` file in the `models` directory.

3. Ensure that the path to this `.bin` file in your code (in `train_model.py`) points to the correct location in the `models` folder.

This step is crucial for the word embedding functionality of the model. Without this file, the model training and prediction processes will not work as intended.
