import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load dataset
df = pd.read_csv("asl_landmarks.csv")
print(f"Dataset shape (raw): {df.shape}")
print(f"Classes (raw): {df['label'].unique()}")
print(f"Samples per class (raw):\n{df['label'].value_counts()}")

# Optionally drop 'nothing' class if it's too underrepresented
MIN_SAMPLES_PER_CLASS = 10
class_counts = df["label"].value_counts()
rare_classes = class_counts[class_counts < MIN_SAMPLES_PER_CLASS].index.tolist()

if rare_classes:
    print(f"\nDropping underrepresented classes (fewer than {MIN_SAMPLES_PER_CLASS} samples): {rare_classes}")
    df = df[~df["label"].isin(rare_classes)].reset_index(drop=True)

print(f"\nDataset shape (filtered): {df.shape}")
print(f"Classes (filtered): {df['label'].unique()}")
print(f"Samples per class (filtered):\n{df['label'].value_counts()}")

# Split features and labels
X = df.drop("label", axis=1)
y = df["label"]

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Model with regularization to reduce overfitting
model = RandomForestClassifier(
    n_estimators=200,  # More trees for better generalization
    max_depth=20,  # Limit depth to prevent overfitting
    min_samples_split=10,  # Require more samples to split
    min_samples_leaf=5,  # Require more samples in leaf nodes
    max_features='sqrt',  # Use sqrt of features (default is 'auto')
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'  # Handle class imbalance if any
)

# Train
print("\nTraining model...")
model.fit(X_train, y_train)

# Cross-validation score (better indicator of real-world performance)
print("\nPerforming cross-validation...")
cv_folds = 5
if class_counts.min() < cv_folds:
    # If any class is still small after filtering, reduce folds
    cv_folds = max(2, int(class_counts.min()))
    print(f"Adjusting CV folds to {cv_folds} due to small class sizes.")

cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
print(f"Cross-validation accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 2 * 100:.2f}%)")

# Evaluate on test set
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest set accuracy: {acc * 100:.2f}%")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Check for overfitting (train vs test accuracy)
train_acc = accuracy_score(y_train, model.predict(X_train))
print(f"\nTrain accuracy: {train_acc * 100:.2f}%")
print(f"Test accuracy: {acc * 100:.2f}%")
print(f"Overfitting gap: {(train_acc - acc) * 100:.2f}%")

if train_acc - acc > 0.15:  # If gap is more than 15%
    print("\n⚠️  WARNING: Model may be overfitting! Consider:")
    print("   - Increasing min_samples_split and min_samples_leaf")
    print("   - Reducing max_depth")
    print("   - Adding more training data")
    print("   - Using data augmentation")

# Save model
joblib.dump(model, "asl_model.pkl")
print("\nModel saved as asl_model.pkl")
