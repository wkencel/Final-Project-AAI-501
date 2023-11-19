import matplotlib.pyplot as plt
import numpy as np
import os
# Sklearn packages
from skimage import io, transform
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder


# Assuming you have downloaded and extracted the dataset locally
data_dir = "../data/trash_images"

# Function to load and preprocess images
def load_images(folder):
    images = []
    # Using folder names as labels
    labels = []
    # Walk directories and files
    for subdir, _, files in os.walk(folder):
        for file in files:
            img_path = os.path.join(subdir, file)
            label = os.path.basename(subdir)
            img = io.imread(img_path)
            # Use 64, 64 to resize images to consistent size, probably something we can play with
            img = transform.resize(img, (64, 64))
            # Part of preprocessing, flatten image
            images.append(img.flatten())
            # Associate labels
            labels.append(label)
    return np.array(images), np.array(labels)


# Function for Random Forest Classifier (output shows about 72% accuracy)
def random_forest(x_train, y_train, x_test, y_test):
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score = accuracy_score(y_pred, y_test)
    # Outputs accuracy scores based on predicted model compared to test data
    print(classification_report(y_pred, y_test))
    confusion_matrix(y_pred, y_test)
    # Call to test what label would be assigned to a random image (expecting cardboard)
    test_with_image(model)
    return score


# Function for K-Nearest Neighbor (output shows about 50% accuracy)
def KNN(x_train, y_train, x_test, y_test):
    # 7 was a number chosen relatively randomly, something we can play with
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(x_train, y_train)
    y_pred_knn = knn.predict(x_test)
    score = accuracy_score(y_pred_knn, y_test)
    # Outputs accuracy scores based on predicted model compared to test data
    print(classification_report(y_pred_knn,y_test))
    confusion_matrix(y_pred_knn, y_test)
    # Call to test what label would be assigned to a random image (expecting cardboard)
    test_with_image(knn)
    return score


# Function for Decision Tree Classifier (output shows about 50% accuracy)
def decision_tree(x_train, y_train, x_test, y_test):
    dtc = DecisionTreeClassifier()
    dtc.fit(x_train, y_train)
    y_pred_dtc = dtc.predict(x_test)
    score = accuracy_score(y_pred_dtc, y_test)
    # Outputs accuracy scores based on predicted model
    print(classification_report(y_pred_dtc, y_test))
    confusion_matrix(y_pred_dtc, y_test)
    # Call to test what label would be assigned to a random image (expecting cardboard)
    test_with_image(dtc)
    return score


# Gaussian Naive Bayes classifier function (output shows about 42% accuracy)
def naive_bayes_classifier(x_train, y_train, x_test, y_test):
    nb = GaussianNB()
    nb.fit(x_train, y_train)
    y_pred_nb = nb.predict(x_test)
    score = accuracy_score(y_pred_nb, y_test)
    print(classification_report(y_pred_nb, y_test))
    confusion_matrix(y_pred_nb, y_test)
    # Call to test what label would be assigned to a random image (expecting cardboard)
    test_with_image(nb)
    return score


# See what labels are given based on each classifier
def test_with_image(classifier):
    # Test with an image
    test_image_path = '../data/trash_images/cardboard/cardboard_015.jpg'
    test_image = io.imread(test_image_path)
    img = transform.resize(test_image, (64, 64))  # Resize images to a consistent size
    img = img.flatten()
    img = img.reshape(1, -1)
    prediction = classifier.predict(img)
    label = label_encoder.inverse_transform(prediction)
    print(f'Using: {classifier}, The predicted class for the given image is: {label}, actual is Cardboard')


# Load images and labels (labels are the names of each folder)
images, labels = load_images(data_dir)

# Encode labels into numerical values
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(images, encoded_labels, test_size=0.2, random_state=42)

# Normalize Data
x_train = x_train/255.0
x_test = x_test/255.0

# Random Forest Call
random_forest_score = random_forest(x_train, y_train, x_test, y_test)

# KNN Call
knn_score = KNN(x_train, y_train, x_test, y_test)

# Decision Tree Call
decision_tree_score = decision_tree(x_train, y_train, x_test, y_test)

# Naive Bayes Call
naive_bayes_score = naive_bayes_classifier(x_train, y_train, x_test, y_test)

# Graph output of scores because we love images!
scores = [random_forest_score, knn_score, decision_tree_score, naive_bayes_score]
labels = ['random_forest', 'knn', 'decision_tree', 'naive_bayes']

plt.bar(labels, scores, color='blue')
plt.xlabel('Algorithm Name')
plt.ylabel('Accuracy Scores')
plt.title('Accuracy Score comparison')

plt.show()



