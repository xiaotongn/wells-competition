import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def encode_labels(y_train):
    le = preprocessing.LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)
    return y_train, le


def cross_validate(x_train, y_train, params, tool='xgboost'):
    y_train, lable_encoder = encode_labels(y_train)

    if tool == 'xgboost':
        dtrain = xgb.DMatrix(x_train, y_train)
        cv_result = xgb.cv(params, dtrain, nfold=10, verbose_eval=True, show_stdv=False)
        return cv_result


def plot_cv(cv_result):
    plt.plot(range(len(cv_result)), cv_result["train-mlogloss-mean"], 'r', label='training loss')
    plt.plot(range(len(cv_result)), cv_result["test-mlogloss-mean"], 'b', label='validation loss')
    plt.legend()
    plt.show()
    return


def plot_learning_curve(x_train, y_train, params):
    y_train, lable_encoder = encode_labels(y_train)
    training_err = []
    validation_err = []
    trainingsize = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    for percent in trainingsize:
        xtrain, xval, ytrain, yval = train_test_split(x_train, y_train, test_size=percent)
        dtrain = xgb.DMatrix(xtrain, ytrain)
        cv_result = xgb.cv(params, dtrain, nfold=5, verbose_eval=False)
        validation_err.append(cv_result["test-mlogloss-mean"].iloc[-1])
        training_err.append(cv_result["train-mlogloss-mean"].iloc[-1])

    plt.plot(trainingsize[::-1], training_err, 'r', label='training error')
    plt.plot(trainingsize[::-1], validation_err, 'g', label='validation error')
    plt.legend()
    plt.xlabel('Training Set size')
    plt.show()
    return


def evaluate(x_train, y_train, params, tool='xgboost', valsize=0.1):
    y_train, label_encoder = encode_labels(y_train)

    if tool == 'xgboost':
        xtrain, xval, ytrain, yval = train_test_split(x_train, y_train, test_size=valsize)
        dtrain = xgb.DMatrix(xtrain, ytrain)
        dval = xgb.DMatrix(xval)
        model = xgb.train(params, dtrain)
        preds = model.predict(dval)
        yval = label_encoder.inverse_transform(yval)
        predictions = pd.DataFrame(columns=['functional', 'functional-need repairs', 'non functional'],
                                   data=preds)
        predictions['validation data'] = yval
        softmax_preds = [np.argmax(x) for x in preds]
        softmax_preds = label_encoder.inverse_transform(softmax_preds)
        predictions['softmax predictions'] = softmax_preds
        accuracy = [1 for i in range(len(predictions)) if \
                    (predictions["validation data"].iloc[i] == predictions["softmax predictions"].iloc[i])]
        accuracy = 1. * np.sum(accuracy) / len(predictions)
        print('\n')
        print("Overall accuracy: %s" %(accuracy))
        print("\n")
        print("Confusion Matrix: \n")
        print(confusion_matrix(predictions["validation data"], predictions["softmax predictions"]))
        print('\n')
    return model

if __name__ == '__main__':
    df = pd.read_csv("training_cleaned.csv", parse_dates=["date_recorded"])
    df = df.fillna(-999)
    to_encode = ['funder', 'installer', 'basin', 'region', 'lga', 'recorded_by', 'scheme_management', 'scheme_name',
                 'extraction_type', 'extraction_type_group', 'payment', 'payment_type', 'water_quality',
                 'quality_group', 'quantity', 'quantity_group', 'source', 'source_type', 'source_class',
                 'waterpoint_type', 'waterpoint_type_group', 'wpt_name', 'subvillage', 'ward',
                 'public_meeting', 'permit', 'extraction_type_class', 'management', 'management_group']
    for col in to_encode:
        le = preprocessing.LabelEncoder()
        le.fit(df[col])
        df[col] = le.transform(df[col])
    
    params = {'eta': 0.2,
              'max_depth': 6,
              'n_estimators': 400,
              'num_boost_round': 200,
              'early_stopping_rounds': 20,
              'objective': 'multi:softprob',
              'num_class': 3,
              'eval_metric': 'mlogloss',
              'silent': 1}

    x_train = df.drop(['id', 'date_recorded','status_group'], axis=1)
    y_train = df['status_group']
    model = evaluate(x_train, y_train, params, tool='xgboost', valsize=0.1)
    plot_learning_curve(x_train, y_train, params)
    cv = cross_validate(x_train, y_train, params, tool='xgboost')
    plot_cv(cv)