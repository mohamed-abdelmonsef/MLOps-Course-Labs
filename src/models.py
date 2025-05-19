from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import logging

def train_random_forest(X_train, y_train):
    logging.info("Training Random Forest model...")
    rf = RandomForestClassifier(n_estimators=100, random_state=6, max_depth=10, max_leaf_nodes=50, criterion='gini') 
    model = rf.fit(X_train, y_train)
    return model

def train_logistic_regression(X_train, y_train):
    logging.info("Training Logistic Regression model...")
    lr = LogisticRegression(random_state=6, max_iter=1000, C=2.0, penalty='l2', solver='liblinear')
    model = lr.fit(X_train, y_train)
    return model

def train_SVC(X_train, y_train):
    logging.info("Training SVC model...")
    svc = SVC(probability=True, kernel='rbf', C=4.0, gamma='scale')
    model = svc.fit(X_train, y_train)
    return model