import numpy as np

class Memory:
    def __init__(self):
        self.data = []

    def add(self, r):
        self.data.append(r)

    def recent(self, k=5):
        return self.data[-k:]
