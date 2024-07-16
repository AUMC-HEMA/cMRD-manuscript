from sklearn.mixture import GaussianMixture
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels

class GMMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components_class0=1, n_components_class1=1, 
                 covariance_type="full", random_state = None):
        self.n_components_class0 = n_components_class0
        self.n_components_class1 = n_components_class1
        self.covariance_type = covariance_type
        self.random_state = None
        
    def fit(self, X, y):
        X_class0 = X[y == 0]
        X_class1 = X[y == 1]
        self.classes_ = unique_labels(y)
        self.gmm_class0 = GaussianMixture(n_components=self.n_components_class0, 
            covariance_type=self.covariance_type, random_state = self.random_state)
        self.gmm_class1 = GaussianMixture(n_components=self.n_components_class1, 
            covariance_type=self.covariance_type, random_state = self.random_state)
        self.gmm_class0 = self.gmm_class0.fit(X_class0)
        self.gmm_class1 = self.gmm_class1.fit(X_class1)

    def predict(self, X):
        prob_class0 = self.gmm_class0.score_samples(X)
        prob_class1 = self.gmm_class1.score_samples(X)
        return (prob_class1 > prob_class0).astype(int)