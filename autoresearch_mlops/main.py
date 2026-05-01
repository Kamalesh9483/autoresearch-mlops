from data import load_data
from core import run

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    run(X_train, X_test, y_train, y_test)
