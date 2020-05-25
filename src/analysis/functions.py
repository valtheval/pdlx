from modelling import ml
import numpy as np
import preprocessing.preprocessing as pp

def bootstrap(model, Xtrain, ytrain, Xtest, ytest, N=10):
    acc_train_vect = []
    acc_test_vect = []
    for i in range(N):
        print("%d / %d " %(i + 1, N))
        model.fit(Xtrain, ytrain)

        dates_pred_train = model.predict(Xtrain)
        acc_train = ml.compute_accuracy(ytrain.values, dates_pred_train)

        dates_pred_test = model.predict(Xtest)
        acc_test = ml.compute_accuracy(ytest.values, dates_pred_test)

        acc_train_vect.append(acc_train)
        acc_test_vect.append(acc_test)
    return acc_train_vect, acc_test_vect
