import numpy as np

class CustomLabelEncoder:
    def __init__(self):
        self.classes_ = None
        self.encoding_mapping_ = None
        self.inverse_mapping_ = None

    def fit(self, y):
        self.classes_ = np.unique(y)
        self.encoding_mapping_ = {label: index for index, label in enumerate(self.classes_)}
        self.inverse_mapping_ = {index: label for label, index in self.encoding_mapping_.items()}
        return self

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y):
        if self.inverse_mapping_ is None:
            raise ValueError("fit() must be called before inverse_transform()")
        return np.vectorize(self.inverse_mapping_.get)(y)

    def transform(self, y):
        if self.encoding_mapping_ is None:
            raise ValueError("fit() must be called before transform()")
        return np.vectorize(self.encoding_mapping_.get)(y)