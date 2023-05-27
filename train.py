import pandas as pd
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib, time, sys

# Load the dataset
data = pd.read_csv('dataset.csv')  # Replace 'your_dataset.csv' with the actual file path

# Load and preprocess the images
X = []
y = []

resizer = int(sys.argv[1])
test_size = float(sys.argv[2])
max_iter = int(sys.argv[3])


print("Image Data Shape: " , data.shape)
print("Image Data Columns: " , data.columns)

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
split_starttime = time.time()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
print('Splitting the data took: ', time.time() - split_starttime, 'seconds')

# Check the dimensions of the training and testing data
print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('X_test shape:', X_test.shape)
print('y_test shape:', y_test.shape)
print('Training samples:', X_train.shape[0])
print('Testing samples:', X_test.shape[0])

# Create the logistic regression model
model = LogisticRegression(max_iter=max_iter)

# Train the model
train_starttime = time.time()
model.fit(X_train, y_train)
print('\nTraining the model took: ', time.time() - train_starttime, 'seconds')

# Evaluate the model, get the accuracy score in percentage between 0-100%
score = model.score(X_test, y_test)
print('Model accuracy:', score * 100, '%\n')


# Predict 1 random image from the test data
random_index = np.random.randint(0, X_test.shape[0])
random_image = X_test[random_index].reshape((1, -1))
random_label = y_test[random_index]
prediction = model.predict(random_image)[0]
print('Predicted label:', prediction)
print('Actual label:', random_label)

# check if score is above 70% if not, run the script again with other parameters
if score >= 0.7:
    print('Model accuracy is above 70%, saving the model weights...')
    # Save the model weights
    joblib.dump(model, 'model_weights_' + str(resizer) + '.joblib')
else:
    print('Model accuracy is below 70%, please run the script again with other parameters')
    sys.exit(1)