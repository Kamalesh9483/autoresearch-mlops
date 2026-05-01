from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd

def load_data():
    """
    Loads a public regression dataset (no local files required)
    """

    # Load dataset from sklearn (downloads once, then cached)
    data = fetch_california_housing()

    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    # Optional: small feature enrichment (safe baseline)
    X["rooms_per_household"] = X["AveRooms"] / (X["AveOccup"] + 1e-5)
    X["bedrooms_ratio"] = X["AveBedrms"] / (X["AveRooms"] + 1e-5)

    return train_test_split(X, y, test_size=0.2, random_state=42)