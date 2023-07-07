from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()

# Split the dataset into features (X) and target labels (y)
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree classifier
clf = DecisionTreeClassifier()

# Train the decision tree classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the accuracy of the model-
accuracy = accuracy_score(y_test, y_pred)
sepallength=input("ENTER SEPAL LENGTH OF FLOWER:")
sepalwidth=input("ENTER SEPAL WIDTH OF FLOWER:")
petallength=input("ENTER PETAL LENGTH OF FLOWER:")
petalwidth=input("ENTER PETAL WIDTH OF FLOWER:")
print("Accuracy:", accuracy)

# Classify a new sample
new_sample = [[sepallength,sepalwidth,petallength,petalwidth]]  # Provide the measurements of the new sample
predicted_class = clf.predict(new_sample)
predicted_species = iris.target_names[predicted_class]
print("Predicted species:", predicted_species)