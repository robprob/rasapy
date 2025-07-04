import numpy as np
from scipy import stats

from rasapy.metrics.classification import accuracy

class NaiveBayesClassification:
    """
    Implementation of a Naive Bayes classifier, assuming strong conditional feature independence given class label.
    Currently supporting Multinomial data (Plans for Gaussian and Bernoulli).
    """
    def __init__(self, distribution='multinomial', alpha=1.0):
        # Validate distribution input
        if distribution in {'multinomial', 'gaussian'}:
            self.distribution = distribution
        else:
            raise ValueError(f"Invalid distribution: {distribution}")
        
        self.alpha = alpha # Laplace smoothing parameter (Additive smoothing)
        
        self.classes = None
        self.priors = None
        
        # Multinomial
        self.class_feature_counts = None
        self.class_feature_freq = None
        
        # Gaussian
        self.means = None
        self.vars = None
        
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
        elif self.distribution == 'gaussian':
            self.fit_gaussian(X_train, y_train)
    
    def fit_multinomial(self, X_train, y_train):
        """
        Fit multinomial data to the Naive Bayes model.
        """
        num_classes, num_features = len(self.classes), X_train.shape[1]
        
        # Calculate total count of feature observations by class label
        class_feature_counts = np.zeros((num_classes, num_features), dtype=int)
        for i, c in enumerate(self.classes):
            # Parse training entries where class label equals current class, sum along features axis (columns)
            class_feature_counts[i] = X_train[y_train == c].sum(axis=0)
        
        # Calculate frequency for each feature as a result of observed counts
        class_feature_freq = np.zeros((num_classes, num_features), dtype=float)
        for i, c in enumerate(self.classes):
            # Perform Laplace smoothing
            # If feature is not seen in any a specific class outputs, it otherwise leads to 0 probability when frequencies are multiplied together
            class_feature_freq[i] = (class_feature_counts[i] + self.alpha) / (np.sum(class_feature_counts[i]) + (self.alpha * num_features))
        
        # Assign class feature counts to the model
        self.class_feature_freq = class_feature_freq
    
    def fit_gaussian(self, X_train, y_train):
        """
        Fit continuous Gaussian data to the Naive Bayes model.
        """
        num_classes, num_features = len(self.classes), X_train.shape[1]
        
        # Calculate class-specific mean and variance for each feature
        means = np.zeros((num_classes, num_features), dtype=float)
        vars = np.zeros((num_classes, num_features), dtype=float)
        
        for i, c in enumerate(self.classes):
            X_class = X_train[y_train == c]
            means[i] = X_class.mean(axis=0)
            vars[i] = np.maximum(np.var(X_class, axis=0), 1e-9) # prevent log(0) later on
        
        # Assign to model
        self.means = means
        self.vars = vars
    
    def predict_proba(self, X):
        """
        Make class probability predictions based on the priors observed in the training data.
        """
        num_classes, num_features = len(self.classes), X.shape[1]
        
        if self.distribution == 'multinomial':
            y_proba = X @ self.class_feature_freq.T # Matrix of size (X.shape[0], num_classes)
        elif self.distribution == 'gaussian':
            y_proba = np.empty((X.shape[0], num_classes), dtype=float)
            for i in range(len(self.classes)):
                # Calculate log of Probability Density Function for a Gaussian distribution
                log_likelihoods = stats.norm.logpdf(X, loc=self.means[i], scale=np.sqrt(self.vars[i])) # takes STD instead of variance
                # Logarithmic product rule allows adding log likelihoods instead of multiplying raw likelihoods (risk of numerical instability/underflow)
                y_proba[:, i] = log_likelihoods.sum(axis=1)
        
        return y_proba
        
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