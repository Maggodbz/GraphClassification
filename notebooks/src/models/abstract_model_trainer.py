from abc import ABC, abstractmethod
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize

class AbstractModelTrainer(ABC):

    def __init__(self, data_path, test_size=0.2, random_state=42):
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        
    def load_data(self):
        self.data_encoded = pickle.load(open(self.data_path, 'rb'))
        self.tfidf_matrix = self.data_encoded.drop(columns=['map_category_encoded']).to_numpy()
        self.target = self.data_encoded['map_category_encoded'].to_numpy()
    
    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.tfidf_matrix, self.target, test_size=self.test_size, random_state=self.random_state)
    
    @abstractmethod
    def initialize_model(self, **kwargs):
        pass
    
    @abstractmethod
    def train_model(self):
        pass

    def evaluate_model(self):
        # Predictions
        train_preds = self.model.predict(self.X_train)
        test_preds = self.model.predict(self.X_test)
        
        # Check if it's a multiclass classification
        is_multiclass = len(set(self.y_train)) > 2

        # Probabilities (required for AUC metrics)
        if hasattr(self.model, "predict_proba"):
            train_probs = self.model.predict_proba(self.X_train)
            test_probs = self.model.predict_proba(self.X_test)
        else:  # Handle models like SVM without probability estimates
            train_probs = None
            test_probs = None

        # Compute metrics
        metrics = {
            "Accuracy": accuracy_score,
            "Precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted' if is_multiclass else 'binary'),
            "Recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted' if is_multiclass else 'binary'),
            "F1-Score": lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted' if is_multiclass else 'binary')
        }

        print("Training Metrics:")
        for name, metric_func in metrics.items():
            print(f"{name}: {metric_func(self.y_train, train_preds):.2f}")

        if train_probs is not None and is_multiclass:
            y_bin_train = label_binarize(self.y_train, classes=list(set(self.y_train)))
            roc_auc = roc_auc_score(y_bin_train, train_probs, multi_class="ovr", average="weighted")
            pr_auc = average_precision_score(y_bin_train, train_probs, average="weighted")
            print(f"AUC-ROC: {roc_auc:.2f}")
            print(f"AUC-PR: {pr_auc:.2f}")
        elif train_probs is not None:
            print(f"AUC-ROC: {roc_auc_score(self.y_train, train_probs[:, 1]):.2f}")
            print(f"AUC-PR: {average_precision_score(self.y_train, train_probs[:, 1]):.2f}")

        print("\nTest Metrics:")
        for name, metric_func in metrics.items():
            print(f"{name}: {metric_func(self.y_test, test_preds):.2f}")

        if test_probs is not None and is_multiclass:
            y_bin_test = label_binarize(self.y_test, classes=list(set(self.y_train)))  # use y_train classes to ensure consistency
            roc_auc = roc_auc_score(y_bin_test, test_probs, multi_class="ovr", average="weighted")
            pr_auc = average_precision_score(y_bin_test, test_probs, average="weighted")
            print(f"AUC-ROC: {roc_auc:.2f}")
            print(f"AUC-PR: {pr_auc:.2f}")
        elif test_probs is not None:
            print(f"AUC-ROC: {roc_auc_score(self.y_test, test_probs[:, 1]):.2f}")
            print(f"AUC-PR: {average_precision_score(self.y_test, test_probs[:, 1]):.2f}")
        
    def run(self, **kwargs):
        self.load_data()
        self.split_data()
        self.initialize_model(**kwargs)
        self.train_model()
        self.evaluate_model()

    def save_model(self, model_path):
        pickle.dump(self.model, open(model_path, 'wb'))