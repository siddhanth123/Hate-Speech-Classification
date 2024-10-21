# Hate Speech Classification Project Walkthrough

## Approach and Thought Process

1. **Data Preparation**:

   - Created a dataset preparation script to combine text files with their annotations.
   - Merged content from individual text files with labels from the annotations file.
   - Saved the prepared datasets as CSV files for easier handling.

2. **Data Preprocessing**:

   - Implemented text preprocessing functions to clean and normalize the text data.
   - Applied lowercase conversion, punctuation removal, and stopword elimination.
   - Utilized the NLTK library for stopword removal to reduce noise in the data.

3. **Tokenization and Padding**:

   - Used Keras Tokenizer to convert text to sequences of integers.
   - Implemented padding to ensure uniform input length for the model.

4. **Word Embeddings**:

   - Leveraged pre-trained Word2Vec embeddings (Google News vectors) to capture semantic meanings.
   - Created an embedding matrix to initialize the model's embedding layer.

5. **Model Architecture**:

   - Designed a sequential model using Keras.
   - Incorporated an Embedding layer initialized with pre-trained weights.
   - Utilized Bidirectional GRU layers to capture context from both directions.
   - Added Dense layers with ReLU activation and Dropout for regularization.
   - Used a final Dense layer with sigmoid activation for binary classification.

6. **Model Training**:

   - Split the data into training and validation sets.
   - Implemented early stopping to prevent overfitting.
   - Trained the model using binary crossentropy loss and Adam optimizer.

7. **Model Evaluation**:

   - Created a separate script for model evaluation on the test set.
   - Generated and saved classification report, confusion matrix, and sample predictions.

8. **API Development**:

   - Implemented a FastAPI server to serve the trained model.
   - Created a POST endpoint for text classification.
   - Ensured proper error handling and input validation.

9. **Documentation**:
   - Provided clear instructions and explanations in the README and project.md files.
   - Included sample CURL commands for API usage.

## API Usage and Testing

There are two main ways to interact with and test the API:

1. **Using CURL Command**:
   You can use the following CURL command to send a request to the API:

   ```bash
   curl -X POST "http://localhost:8080/predict" \
        -H "Content-Type: application/json" \
        -d '{\"text\": \"This is a sample text to classify\"}'
   ```

   Note: For Windows Command Prompt, use double quotes for the entire command and escape the inner double quotes.

   This command sends a POST request to the `/predict` endpoint with a JSON payload containing the text to be classified. The API will return a JSON response with the prediction (hate or noHate) and the probability.

2. **Using Swagger UI**:
   FastAPI provides an interactive Swagger UI for easy testing and exploration of the API:

   - Run the `app.py` file to start the FastAPI server:
     ```
     python app.py
     ```
   - Open a web browser and navigate to `http://127.0.0.1:8080/docs`
   - You'll see the Swagger UI interface with your API endpoints.
   - Click on the `/predict` endpoint, then click "Try it out".
   - Enter your text in the request body and click "Execute".
   - The API response will be displayed directly in the browser.

   This method provides a user-friendly interface for testing the API without needing to use command-line tools.

Both methods allow you to send text inputs to the API and receive predictions, making it easy to test and interact with your hate speech classification model.

## Reflection

This approach combines traditional NLP techniques with modern deep learning methods. The use of pre-trained word embeddings helps in capturing semantic relationships, while the bidirectional GRU layers allow the model to understand context from both directions in the text. The modular structure of the code allows for easy modifications and improvements in the future.

Areas for potential improvement include:

- Experimenting with different model architectures (e.g., LSTM, Transformer-based models).
- Implementing cross-validation for more robust evaluation.
- Exploring advanced text preprocessing techniques like lemmatization or bag of words.
- Implementing data augmentation to handle class imbalance if present.

Overall, this project demonstrates a comprehensive approach to text classification, from data preparation to model deployment, suitable for production environments.

## Project Structure

For a detailed overview of the project structure and file descriptions, please refer to the [Project Structure Document](./project_structure.md).

## Important Note on Word2Vec Embeddings

For the sake of keeping the project size manageable, the Google News Word2Vec file has been removed from this repository. To run the entire project successfully, you need to download and add this file:

For more information, please refer to the [Important Note on Word2Vec Embeddings](./project_structure.md).
