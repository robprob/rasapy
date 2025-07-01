import numpy as np

from rasapy.metrics.classification import accuracy

class NaiveBayesClassification:
    """
    Implementation of a Naive Bayes classifier, assuming strong conditional feature independence given class label.
    Currently supporting Multinomial data (Plans for Gaussian and Bernoulli).
    """
    def __init__(self, distribution='multinomial'):
        # Validate distribution input
        if distribution in {'multinomial'}:
            self.distribution = distribution
        else:
            raise ValueError(f"Invalid distribution: {distribution}")
        
        self.classes = None
        self.priors = None
        self.class_feature_counts = None
        
    def fit(self, X_train, y_train):
        """
        Fit Naive Bayes model to the training data using the specified data distribution/model variation.
        """
        # Parse output class labels and their frequency counts in the training data
        self.classes, class_counts = np.unique(y_train, return_counts=True)
        
        # Calculate class priors based on observed proportional frequency (Maximum Likelihood Estimation)
        self.priors = class_counts / np.sum(class_counts)
        
        # Delegate to distribution-specific model fit function
        if self.distribution == 'multinomial':
            self.fit_multinomial(X_train, y_train)
    
    def fit_multinomial(self, X_train, y_train):
        """
        Fit multinomial data to the Naive Bayes model.
        """
        # Calculate total count of feature observations by class label
        class_feature_counts = np.zeros((len(self.classes), X_train.shape[1]), dtype=float)
        
        for i, c in enumerate(self.classes):
            # Parse training entries where class label equals current class, sum along features axis (columns)
            class_feature_counts[i] = X_train[y_train == c].sum(axis=0)
        
        # Calculate frequency for each feature as a result of observed counts
        for i, c in enumerate(self.classes):
            class_feature_counts[i] /= np.sum(class_feature_counts[i])
        
        # Assign class feature counts to the model
        self.class_feature_counts = class_feature_counts
    
    def predict_proba(self, X):
        """
        Make class probability predictions based on the priors observed in the training data.
        """
        return (X @ self.class_feature_counts.T) # Matrix of size (X.shape[0], num_classes)
        
    def predict(self, X):
        """
        Make class predictions based on the maximum predicted probability.
        """
        return self.classes[np.argmax(self.predict_proba(X), axis=1)]
    
    def score(self, X, y_true):
        """
        Make classification predictions and calculate accuracy.
        """
        y_pred = self.predict(X)
        return accuracy(y_true, y_pred)