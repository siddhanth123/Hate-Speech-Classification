import os
import pandas as pd
from pathlib import Path

def create_dataset(train_folder, annotations_file):
    """
    Create a dataset by combining text files with their annotations.

    This function reads text files from a specified folder and combines their
    content with corresponding labels from an annotations file. It creates a
    pandas DataFrame with columns for file_id, content, and label.

    Parameters:
    train_folder (str or Path): Path to the folder containing text files.
    annotations_file (str or Path): Path to the CSV file containing annotations.

    Returns:
    pandas.DataFrame: A DataFrame with columns 'file_id', 'content', and 'label'.

    Note:
    - Text files in the train_folder should have a .txt extension.
    - The annotations_file should be a CSV with at least 'file_id' and 'label' columns.
    - File IDs in the annotations file should match the text file names (without extension).
    """
    # Read the annotations file
    annotations = pd.read_csv(annotations_file)
    
    # Create an empty list to store our dataset
    dataset = []
    
    # Iterate through all files in the train folder
    for filename in os.listdir(train_folder):
        if filename.endswith('.txt'):
            file_id = filename.split('.')[0]  # Get the file ID without extension
            
            # Read the content of the text file
            with open(os.path.join(train_folder, filename), 'r', encoding='utf-8') as file:
                content = file.read().strip()
            
            # Find the corresponding label in the annotations
            label = annotations.loc[annotations['file_id'] == file_id, 'label'].values
            
            if len(label) > 0:
                # Add the content and label to our dataset
                dataset.append({
                    'file_id': file_id,
                    'content': content,
                    'label': label[0]
                })
    
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(dataset)
    
    return df

train_folder = Path('data/sampled_train')
test_folder = Path('data/sampled_test')

annotations_file = Path('data/annotations_metadata.csv')
print(annotations_file)

train_dataset = create_dataset(train_folder, annotations_file)
test_dataset = create_dataset(test_folder, annotations_file)

# Display the first few rows of the dataset
print(train_dataset.head())
print(test_dataset.head())

# Save the dataset to a CSV file
train_dataset.to_csv('data/modified_dataset/hate_speech_train_dataset.csv', index=False)
test_dataset.to_csv('data/modified_dataset/hate_speech_test_dataset.csv', index=False)

