from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from .abstract_model_trainer import AbstractModelTrainer

class ModelFactory:
    @staticmethod
    def get_model_trainer(model_type, data_path, **kwargs):
        if model_type == "logistic_regression":
            return LogisticRegressionTrainer(data_path, **kwargs)
        elif model_type == "decision_tree":
            return DecisionTreeTrainer(data_path, **kwargs)
        elif model_type == "random_forest":
            return RandomForestTrainer(data_path, **kwargs)
        elif model_type == "svc":
            return SVCTrainer(data_path, **kwargs)
        elif model_type == "knn":
            return KNNTrainer(data_path, **kwargs)
        elif model_type == "naive_bayes":
            return NaiveBayesTrainer(data_path, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


class LogisticRegressionTrainer(AbstractModelTrainer):
    def initialize_model(self, **kwargs):
        self.model = LogisticRegression(**kwargs)
    
    def train_model(self):
        self.model.fit(self.X_train, self.y_train)
        

class DecisionTreeTrainer(AbstractModelTrainer):
    def initialize_model(self, **kwargs):
        self.model = DecisionTreeClassifier(**kwargs)
    
    def train_model(self):
        self.model.fit(self.X_train, self.y_train)


class RandomForestTrainer(AbstractModelTrainer):
    def initialize_model(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs)
    
    def train_model(self):
        self.model.fit(self.X_train, self.y_train)


class SVCTrainer(AbstractModelTrainer):
    def initialize_model(self, **kwargs):
        self.model = SVC(**kwargs)
    
    def train_model(self):
        self.model.fit(self.X_train, self.y_train)


class KNNTrainer(AbstractModelTrainer):
    def initialize_model(self, **kwargs):
        self.model = KNeighborsClassifier(**kwargs)
    
    def train_model(self):
        self.model.fit(self.X_train, self.y_train)


class NaiveBayesTrainer(AbstractModelTrainer):
    def initialize_model(self, **kwargs):
        self.model = MultinomialNB(**kwargs)
    
    def train_model(self):
        self.model.fit(self.X_train, self.y_train)


