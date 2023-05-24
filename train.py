import pandas as pd
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Load the dataset
data = pd.read_csv('dataset.csv')  # Replace 'your_dataset.csv' with the actual file path

# Load and preprocess the images
X = []
y = []

resizer = 1024

for index, row in data.iterrows():
    image = Image.open(row['image_path'])  # Replace 'image_path' with the column name containing the image file paths
    image = image.resize((resizer, resizer))  # Resize the image to a desired size
    image = np.array(image)
    X.append(image)
    y.append(row['label'])  # Replace 'label' with the column name containing the image labels

X = np.stack(X)
y = np.array(y)

# Flatten the image data
n_samples = X.shape[0]
X = X.reshape((n_samples, -1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Check the dimensions of the training and testing data
print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('X_test shape:', X_test.shape)
print('y_test shape:', y_test.shape)
print('Training samples:', X_train.shape[0])
print('Testing samples:', X_test.shape[0])

# Create the logistic regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
score = model.score(X_test, y_test)
print('Model accuracy:', score, '%')

# Save the model weights
joblib.dump(model, 'model_weights_' + str(resizer) + '.joblib')