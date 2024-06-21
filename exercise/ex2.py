def homework():
    "Create a logistic regression model for the breast cancer dataset."
    "lg: a logistic regression model"
    "R2: the R2 score of the model"
    "Complete the function"

    import numpy as np
    import pandas as pd

    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X = data.data
    y = data.target
    from sklearn.linear_model import LogisticRegression
    lg = LogisticRegression(max_iter= 10000, penalty = None, solver ="newton-cg")
    R2 = None
    
    
    return  lg, R2
 

if __name__ == '__main__':
    print(homework())
