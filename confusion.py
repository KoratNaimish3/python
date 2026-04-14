# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix, classification_report

# # Step 1: Create dataset
# X = np.array([[1],[2],[3],[4],[5],[6]])
# y = np.array([0,0,0,1,1,1])

# # Step 2: Split dataset
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.33, random_state=42)


# # Step 3: Train model
# model = LogisticRegression()
# model.fit(X_train, y_train)

# # Step 4: Predict results
# y_pred = model.predict(X_test)

# # Step 5: Create confusion matrix
# cm = confusion_matrix(y_test, y_pred, labels=[0,1])

# cm = confusion_matrix(y_test, y_pred)

# print("Confusion Matrix:\n", cm)

# # Step 6: Plot heatmap
# sns.heatmap(cm, annot=True, fmt='d')
# plt.title("Confusion Matrix Heatmap")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.show()

# # Step 7: Performance report
# print("\nClassification Report:\n")
# print(classification_report(y_test, y_pred))


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

y_true = [0,0,1,1]
y_pred = [0,1,1,1]

cm = confusion_matrix(y_true, y_pred)

print("Confusion Matrix:")
print(cm)

sns.heatmap(cm, annot=True, fmt='d')

plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()