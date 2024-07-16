import flowsom as fs
import anndata as ad
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels

class FlowSOMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, xdim = 10, ydim = 10, ratio_threshold = 2, random_state = None):
        self.xdim = xdim
        self.ydim = ydim
        self.ratio_threshold = ratio_threshold
        self.random_state = random_state
        self.clusters = None
        
    def fit(self, X, y):
        self.classes_ = unique_labels(y)
        # Perform FlowSOM clustering
        self.SOM = fs.FlowSOM(ad.AnnData(X), cols_to_use=list(X.columns), 
                              n_clusters=1, xdim=self.xdim, ydim=self.ydim, 
                              seed=self.random_state)
        # Get the clusters associated with positive class
        # Get the cluster counts by class
        df = pd.DataFrame({"cluster": list(self.SOM.cluster_labels), "class": y})
        counts = df.groupby(["cluster", "class"]).size().unstack(fill_value=0)
        # Convert to percentages
        percentages = counts.copy()
        percentages[0] = percentages[0] / percentages[0].sum() * 100
        percentages[1] = percentages[1] / percentages[1].sum() * 100
        # Calculate the ratio
        # Add a very small number to deal with zeroes
        percentages["ratio"] = percentages[1] / (percentages[0] + 1e-6)
        # Get the clusters
        self.clusters = percentages[percentages["ratio"] > self.ratio_threshold].index.tolist()

    def predict(self, X):
        # Map new data and predict based on clusters
        newSOM = self.SOM.new_data(ad.AnnData(X))
        return np.array([1 if i in self.clusters else 0 for i in newSOM.cluster_labels])
