import os
import numpy as np
from skimage import io, transform
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Assuming you have downloaded and extracted the dataset locally
data_dir = "../data/trash_images"

# Function to load and preprocess images
def load_images(folder):
    images = []
    labels = []
    for subdir, _, files in os.walk(folder):
        for file in files:
            img_path = os.path.join(subdir, file)
            label = os.path.basename(subdir)
            img = io.imread(img_path)
            img = transform.resize(img, (64, 64))  # Resize images to a consistent size
            images.append(img.flatten())  # Flatten the image
            labels.append(label)
    return np.array(images), np.array(labels)

# Load images and labels
images, labels = load_images(data_dir)

# Encode labels into numerical values
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, encoded_labels, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Test with an image
test_image_path = '../data/kim_test_images/my_glass1.jpg'
test_image = io.imread(test_image_path)
img = transform.resize(test_image, (64, 64))  # Resize images to a consistent size
img = img.flatten()
img = img.reshape(1, -1)
prediction = rf_classifier.predict(img)
label = label_encoder.inverse_transform(prediction)
print(f'The predicted class for the given image is: {label}')
