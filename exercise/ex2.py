def homework():
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X = data.data
    y = data.target
    from sklearn.linear_model import LogisticRegression
    lg = LogisticRegression(max_iter= 10000, penalty = None, solver ="newton-cg", random_state=0)
    lg.fit(X, y)
    y_pred = lg.predict(X)
    R2 = lg.score(X, y)
    print("R2: ", R2)
    print("coef: ", lg.coef_[0,0])
    print("intercept: ", lg.intercept_)
    print("predict: ", lg.predict(X[0:1]))
    print('X:', X[0:1])
    return  lg, R2
 

if __name__ == '__main__':
    print(homework())
