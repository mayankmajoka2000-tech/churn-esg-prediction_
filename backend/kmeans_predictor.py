class ScaledKMeansPredictor:
    """K-Means predictor that scales raw segmentation variables before prediction."""

    def __init__(self, scaler, kmeans):
        self.scaler = scaler
        self.kmeans = kmeans

    def predict(self, X):
        return self.kmeans.predict(self.scaler.transform(X))
