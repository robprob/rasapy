import numpy as np

from rasapy.metrics.classification import accuracy

class NaiveBayesClassification:
    """
    Implementation of a Naive Bayes classifier, assuming strong conditional feature independence given class label.
    Currently supporting Multinomial data (Plans for Gaussian and Bernoulli).
    """
    def __init__(self, distribution='multinomial', alpha=1.0):
        # Validate distribution input
        if distribution in {'multinomial'}:
            self.distribution = distribution
        else:
            raise ValueError(f"Invalid distribution: {distribution}")
        
        self.alpha = alpha # Laplace smoothing parameter (Additive smoothing)
        
        self.classes = None
        self.priors = None
        self.class_feature_counts = None
        self.class_feature_freq = None
        
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
        num_features = X_train.shape[1]
        
        # Calculate total count of feature observations by class label
        class_feature_counts = np.zeros((len(self.classes), num_features), dtype=int)
        for i, c in enumerate(self.classes):
            # Parse training entries where class label equals current class, sum along features axis (columns)
            class_feature_counts[i] = X_train[y_train == c].sum(axis=0)
        
        # Calculate frequency for each feature as a result of observed counts
        class_feature_freq = np.zeros((len(self.classes), X_train.shape[1]), dtype=float)
        for i, c in enumerate(self.classes):
            # Perform Laplace smoothing
            # If feature is not seen in any a specific class outputs, it otherwise leads to 0 probability when frequencies are multiplied together
            class_feature_freq[i] = (class_feature_counts[i] + self.alpha) / (np.sum(class_feature_counts[i]) + (self.alpha * num_features))
        
        # Assign class feature counts to the model
        self.class_feature_freq = class_feature_freq
    
    def predict_proba(self, X):
        """
        Make class probability predictions based on the priors observed in the training data.
        """
        return (X @ self.class_feature_freq.T) # Matrix of size (X.shape[0], num_classes)
        
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