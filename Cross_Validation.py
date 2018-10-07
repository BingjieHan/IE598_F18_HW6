import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

#training set & test set
a=[]
b=[]
iris = datasets.load_iris()
X = iris.data
y = iris.target

for r in range(1,11):
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=r, stratify=y)
    tree = DecisionTreeClassifier(criterion='gini', 
                              max_depth=4, 
                              random_state=1)
    tree.fit(X_train, y_train)
    y_ptrain=tree.predict(X_train)
    y_ptest=tree.predict(X_test)
    a.append(accuracy_score(y_train, y_ptrain))
    b.append(accuracy_score(y_test,y_ptest)) 
print(np.mean(a),np.mean(b),np.std(a),np.std(b))

#Cross Validation with 10 Folds
acc_r=[]
score=[]
for rr in range(1,11):
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=rr, stratify=y)
    tree_cross = DecisionTreeClassifier(criterion='gini', 
                              max_depth=4, 
                              random_state=1)
    scores = cross_val_score(estimator=tree_cross, X=X_train, y=y_train, cv=10, n_jobs=1)
    acc_r.append(np.mean(scores))
    tree_cross.fit(X_train, y_train)
    y_pred=tree_cross.predict(X_test)
    score.append(accuracy_score(y_test, y_pred))

print(acc_r)
print(score)
print(np.std(acc_r))
print(np.std(score))

print("My name is Bingjie Han")
print("My NetID is: bingjie5")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

