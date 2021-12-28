from functools import wraps

class Explanation:
    explainers = {}
    def __init__(self, explanation_type):
        self.explanation_type = explanation_type
        if explanation_type not in self.explainers:
            self.explainers[explanation_type] = []

    def __call__(self, function):
        self.explainers[self.explanation_type].append(function)
        @wraps(function)
        def wrapper(*args, **kwargs):
            return function(*args, **kwargs)
        return wrapper
