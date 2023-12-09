import os
import numpy as np
from skimage import io, transform
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.preprocessing import LabelEncoder
import pandas as pd

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
rf_classifier = RandomForestClassifier(n_estimators=1000, random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Test with an image
test_image_path = '../data/trash_images/glass/glass_003.jpg'
test_image = io.imread(test_image_path)
img = transform.resize(test_image, (64, 64))  # Resize images to a consistent size
img = img.flatten()
img = img.reshape(1, -1)
prediction = rf_classifier.predict(img)
label = label_encoder.inverse_transform(prediction)
print(f'The predicted class for the given image is: {label}')

# here is the code to get the metrics and export them to a csv

# Evaluate additional metrics
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
conf_matrix = confusion_matrix(y_test, y_pred)

#  "macro-average" refers to the method of first calculating the metric independently for each class 
# and then taking the average of those values. 
# This means that each class is given equal weight, 
# regardless of its prevalence in the dataset.

print(f"Precision (Macro-average): {precision:.4f}")
print(f"Recall (Macro-average): {recall:.4f}")
print(f"F1 Score (Macro-average): {f1:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")

# Calculate per-class accuracies
per_class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

# Since we don't have a loss metric, we can ignore it or use a placeholder like NaN
loss = np.nan

# Now combine everything into a single record for CSV export
headers = ['model', 'loss', 'accuracy', 'precision', 'f1_score'] + \
          [f'accuracy_for_{cls}' for cls in label_encoder.classes_]

record = ['RandomForestClassifier', loss, accuracy, precision, f1] + per_class_accuracies.tolist()

# Save to CSV
metrics_file = '../src/model_metrics.csv'
if not os.path.isfile(metrics_file):
    # If the CSV doesn't exist, create it and write the header
    df = pd.DataFrame(columns=headers)
    df.loc[0] = record
    df.to_csv(metrics_file, index=False)
else:
    # If it does, append the new record
    df = pd.read_csv(metrics_file)
    new_row = pd.Series(record, index=headers)
    new_row_df = pd.DataFrame([record], columns=headers)
    df = pd.concat([df, new_row_df], ignore_index=True)
    df.to_csv(metrics_file, index=False)

print(f"Metrics saved to {metrics_file}")