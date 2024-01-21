import math
from collections import Counter
from typing import List, Union

class TFIDF:
    def __init__(self, max_features=None):
        self.max_features = max_features
        self.documents = []
        self.vocabulary = {}
        self.idf = {}

    def fit(self, raw_documents: List[Union[str, bytes]]):
        # Tokenize and build term frequency for each document
        term_frequencies = []
        for doc in raw_documents:
            tokens = self.tokenize(doc)
            term_freq = Counter(tokens)
            term_frequencies.append(term_freq)

        # Build vocabulary based on document frequencies
        doc_count = len(raw_documents)
        for term_freq in term_frequencies:
            for term in term_freq:
                if term in self.vocabulary:
                    self.vocabulary[term] += 1
                else:
                    self.vocabulary[term] = 1

        # Sort vocabulary by frequency and apply max_features
        sorted_vocab = sorted(self.vocabulary.items(), key=lambda x: x[1], reverse=True)
        if self.max_features is not None:
            sorted_vocab = sorted_vocab[:self.max_features]
        self.vocabulary = dict(sorted_vocab)

        # Calculate inverse document frequency (IDF)
        for term in self.vocabulary:
            doc_freq = sum(1 for term_freq in term_frequencies if term in term_freq)
            self.idf[term] = math.log(doc_count / (1 + doc_freq))

    def transform(self, raw_documents: List[Union[str, bytes]]) -> List[List[float]]:
        # Transform documents to TF-IDF matrix
        tfidf_matrix = []
        for doc in raw_documents:
            tokens = self.tokenize(doc)
            term_freq = Counter(tokens)
            tfidf_vector = [self.calculate_tfidf(term, term_freq) for term in self.vocabulary]
            tfidf_matrix.append(tfidf_vector)

        return tfidf_matrix

    def calculate_tfidf(self, term: str, term_freq: Counter) -> float:
        # Calculate TF-IDF for a specific term in a document
        term_count = term_freq[term]

        # Handle the case where term has zero frequency
        if term_count == 0:
            return 0.0

        tf = term_count / sum(term_freq.values())
        idf = self.idf.get(term, 0.0)
        return tf * idf


    def tokenize(self, document: Union[str, bytes]) -> List[str]:
        # Simple tokenization function
        if isinstance(document, bytes):
            document = document.decode('utf-8')
        return document.lower().split()

    def get_feature_names_out(self) -> List[str]:
        # Get feature names corresponding to the columns of the TF-IDF matrix
        return list(self.vocabulary.keys())

