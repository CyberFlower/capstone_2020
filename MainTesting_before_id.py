import os
import csv
from Utils import Message, CURRENT_FOLDER, plot_confusion_matrix

import pandas as pd # for data analytics
import numpy as np # for numerical computation
from matplotlib import pyplot as plt, style # for ploting
from sklearn.metrics import accuracy_score, fbeta_score, precision_score, recall_score, confusion_matrix # for evaluation
from scipy.stats import multivariate_normal
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import VotingClassifier

style.use('ggplot')
np.random.seed(31)

training_packet = {
    'timestamp': [],
    'canid': [],
    'before_canid': [],
    'after_canid': [],
    'datalen': [],
    'data': [],
    'flag': [],
    # 'before_flag': []
}
testing_packet = {
    'timestamp': [],
    'canid': [],
    'before_canid': [],
    'after_canid': [],
    'datalen': [],
    'data': [],
    'flag': [],
    # 'before_flag': []
}


# read csv file by path, filename
# code for save data will be add later
def read_file(path, filename, packet):
    with open(os.path.join(path, filename), 'r') as fp:
        reader = csv.reader(fp)
        cnt = 0
        before_timestamp = 0
        before_id = 0

        for row in reader:
            if len(row) == 0:
                print(" [-] debugging... line " + str(cnt) + " in  " + filename + " is empty")

            elif row[-1] == 'R' or row[-1] == 'T':
                if cnt == 0:
                    before_timestamp = 0
                    # before_flag = 0

                datalen = int(row[2])

                msg = []
                if row[3].count(" "):
                    msg = row[3].split(" ")
                else:
                    msg = row[3:-1]

                data = 0
                for idx in range(datalen):
                    data *= 0x100
                    data += int(msg[idx], 16)

                timestamp = (float(row[0]) - before_timestamp) * 1000000.0

                canid = int(row[1], 16)
                before_canid = before_id
                before_id = canid

                if row[-1] == 'R':
                    flag = 0
                elif row[-1] == 'T':
                    flag = 1
                else:
                    print(row[-1])

                packet['timestamp'].append(timestamp)
                packet['canid'].append(canid)
                packet['before_canid'].append(before_canid)
                packet['datalen'].append(datalen)
                packet['data'].append(data)
                packet['flag'].append(flag)
                # packet['before_flag'].append(before_flag)

                before_timestamp = float(row[0])
                # before_flag = flag

                if cnt != 0:
                    packet['after_canid'].append(canid) # 두 번째 줄부터 after_canid 를 추가하면 자동으로 순서가 맞춰짐

                cnt += 1

            else:
                exit("csv read error")

        packet['after_canid'].append(0) # 마지막 메시지의 after_canid 는 0


def read_csv_train(car_type, attack_type):
    """ params: car_type, attack_type """
    path = os.path.join(CURRENT_FOLDER, "study_input", car_type)
    file_list = os.listdir(path)
    for file in file_list:
        if attack_type in file:
            read_file(path, file, training_packet)
            break


def read_csv_test(car_type, attack_type):
    """ params: car_type, attack_type """
    path = os.path.join(CURRENT_FOLDER, "test_input", car_type)
    file_list = os.listdir(path)
    for file in file_list:
        if attack_type in file:
            read_file(path, file, testing_packet)
            break


def clear_dict(my_dict):
    """ 
    param: dict
    clear items from dictionary, but remain keys
    if you want to delete all (include keys), do it just my_dict.clear() """  
    for key in my_dict.keys():
        my_dict[key].clear()


def statistical_tec(train, valid, test, car, attack):
    mu = train.drop('flag', axis=1).mean(axis=0).values
    sigma = train.drop('flag', axis=1).cov().values
    model = multivariate_normal(cov=sigma, mean=mu, allow_singular=True)

    # print(np.median(model.logpdf(valid[valid['flag'] == 0].drop('flag', axis=1).values)))
    # print(np.median(model.logpdf(valid[valid['flag'] == 1].drop('flag', axis=1).values)))

    tresholds = np.linspace(-45, -44, 1000)
    scores = []
    for treshold in tresholds:
        y_hat = (model.logpdf(valid.drop('flag', axis=1).values) < treshold).astype(int)
        scores.append([recall_score(y_pred=y_hat, y_true=valid['flag'].values),
                       precision_score(y_pred=y_hat, y_true=valid['flag'].values),
                       fbeta_score(y_pred=y_hat, y_true=valid['flag'].values, beta=1)])
    scores = np.array(scores)

    final_tresh = tresholds[scores[:, 2].argmax()]
    y_hat_test = (model.logpdf(test.drop('flag', axis=1).values) < final_tresh).astype(int)

    # print('Final threshold: %.8f' % final_tresh)
    print('Accuracy: {:.4f}%'.format(accuracy_score(test['flag'], y_hat_test) * 100))
    print('Recall: {:.4f}%'.format(recall_score(test['flag'], y_hat_test) * 100))
    print('Precision: {:.4f}%'.format(precision_score(test['flag'], y_hat_test) * 100))
    print('F1 Score: {:.4f}%'.format(fbeta_score(test['flag'], y_hat_test, beta=1) * 100))

    cnf_matrix = confusion_matrix(test['flag'].values, y_hat_test)
    plot_confusion_matrix(cnf_matrix, classes=['Normal', 'Abnormal'], title='Confusion matrix', car=car, attack=attack)


def linSVM_tec(train, valid, test, car, attack):
    train_X = valid.drop(['flag'], axis=1)
    train_Y = valid['flag']
    test_X = test.drop(['flag'], axis=1)
    test_Y = test['flag']

    model = svm.SVC(kernel='linear', C=0.1, gamma=0.1)
    model.fit(train_X, train_Y)
    preds = model.predict(test_X)

    print('Accuracy: {:.4f}%'.format(accuracy_score(test_Y, preds) * 100))
    print('Recall: {:.4f}%'.format(recall_score(test_Y, preds) * 100))
    print('Precision: {:.4f}%'.format(precision_score(test_Y, preds) * 100))
    print('F1 score: {:.4f}%'.format(fbeta_score(test_Y, preds, beta=1) * 100))

    cnf_matrix = confusion_matrix(test_Y, preds)
    plot_confusion_matrix(cnf_matrix, classes=['Normal', 'Abnormal'], title='Confusion matrix', car=car, attack=attack)


def rbfSVM_tec(train, valid, test, car, attack):
    train_X = valid.drop(['flag'], axis=1)
    train_Y = valid['flag']
    test_X = test.drop(['flag'], axis=1)
    test_Y = test['flag']

    model = svm.SVC(kernel='rbf', C=10, gamma=10)
    model.fit(train_X, train_Y)
    preds = model.predict(test_X)

    print('Accuracy: {:.4f}%'.format(accuracy_score(test_Y, preds) * 100))
    print('Recall: {:.4f}%'.format(recall_score(test_Y, preds) * 100))
    print('Precision: {:.4f}%'.format(precision_score(test_Y, preds) * 100))
    print('F1 score: {:.4f}%'.format(fbeta_score(test_Y, preds, beta=1) * 100))

    cnf_matrix = confusion_matrix(test_Y, preds)
    plot_confusion_matrix(cnf_matrix, classes=['Normal', 'Abnormal'], title='Confusion matrix', car=car, attack=attack)


def gaussian_nb(train, valid, test, car, attack):
    train_X = valid.drop(['flag'], axis=1)
    train_Y = valid['flag']
    test_X = test.drop(['flag'], axis=1)
    test_Y = test['flag']

    model = GaussianNB()
    model.fit(train_X, train_Y)
    preds = model.predict(test_X)

    print('Accuracy: {:.4f}%'.format(accuracy_score(test_Y, preds) * 100))
    print('Recall: {:.4f}%'.format(recall_score(test_Y, preds) * 100))
    print('Precision: {:.4f}%'.format(precision_score(test_Y, preds) * 100))
    print('F1 score: {:.4f}%'.format(fbeta_score(test_Y, preds, beta=1) * 100))

    cnf_matrix = confusion_matrix(test_Y, preds)
    plot_confusion_matrix(cnf_matrix, classes=['Normal', 'Abnormal'], title='Confusion matrix', car=car, attack=attack)


def decisiontree_tec(train, valid, test, car, attack):
    train_X = valid.drop(['flag'], axis=1)
    train_Y = valid['flag']
    test_X = test.drop(['flag'], axis=1)
    test_Y = test['flag']

    clf = DecisionTreeClassifier(max_depth=8, random_state=0)
    clf.fit(train_X, train_Y)

    preds = clf.predict(test_X)

    print('Accuracy: {:.4f}%'.format(accuracy_score(test_Y, preds) * 100))
    print('Recall: {:.4f}%'.format(recall_score(test_Y, preds) * 100))
    print('Precision: {:.4f}%'.format(precision_score(test_Y, preds) * 100))
    print('F1 score: {:.4f}%'.format(fbeta_score(test_Y, preds, beta=1) * 100))

    cnf_matrix = confusion_matrix(test_Y, preds)
    plot_confusion_matrix(cnf_matrix, classes=['Normal', 'Abnormal'], title='Confusion matrix', car=car, attack=attack)


def randomforest_tec(train, valid, test, car, attack):
    train_X = valid.drop(['flag'], axis=1)
    train_Y = valid['flag']
    test_X = test.drop(['flag'], axis=1)
    test_Y = test['flag']

    model = RandomForestClassifier(n_estimators=10)
    model.fit(train_X, train_Y)
    preds = model.predict(test_X)

    print('Accuracy: {:.4f}%'.format(accuracy_score(test_Y, preds) * 100))
    print('Recall: {:.4f}%'.format(recall_score(test_Y, preds) * 100))
    print('Precision: {:.4f}%'.format(precision_score(test_Y, preds) * 100))
    print('F1 score: {:.4f}%'.format(fbeta_score(test_Y, preds, beta=1) * 100))

    cnf_matrix = confusion_matrix(test_Y, preds)
    plot_confusion_matrix(cnf_matrix, classes=['Normal', 'Abnormal'], title='Confusion matrix', car=car, attack=attack)


def kNN_tec(train, valid, test, car, attack):
    train_X = valid.drop(['flag'], axis=1)
    train_Y = valid['flag']
    test_X = test.drop(['flag'], axis=1)
    test_Y = test['flag']

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(train_X, train_Y)
    preds = model.predict(test_X)

    print('Accuracy: {:.4f}%'.format(accuracy_score(test_Y, preds) * 100))
    print('Recall: {:.4f}%'.format(recall_score(test_Y, preds) * 100))
    print('Precision: {:.4f}%'.format(precision_score(test_Y, preds) * 100))
    print('F1 score: {:.4f}%'.format(fbeta_score(test_Y, preds, beta=1) * 100))

    cnf_matrix = confusion_matrix(test_Y, preds)
    plot_confusion_matrix(cnf_matrix, classes=['Normal', 'Abnormal'], title='Confusion matrix', car=car, attack=attack)


def voting_classifier(train, valid, test, car, attack):
    train_X = valid.drop(['flag'], axis=1)
    train_Y = valid['flag']
    test_X = test.drop(['flag'], axis=1)
    test_Y = test['flag']

    kNN = KNeighborsClassifier(n_neighbors=4)
    RF = RandomForestClassifier(n_estimators=10)
    DT = DecisionTreeClassifier(max_depth=6, random_state=0)

    model = VotingClassifier(estimators=[('kNN', kNN), ('RandomForest', RF), ('DecisionTree', DT)], voting='soft', weights=[1.5, 1, 1])
    model.fit(train_X, train_Y)
    preds = model.predict(test_X)

    print('Accuracy: {:.4f}%'.format(accuracy_score(test_Y, preds) * 100))
    print('Recall: {:.4f}%'.format(recall_score(test_Y, preds) * 100))
    print('Precision: {:.4f}%'.format(precision_score(test_Y, preds) * 100))
    print('F1 score: {:.4f}%'.format(fbeta_score(test_Y, preds, beta=1) * 100))

    cnf_matrix = confusion_matrix(test_Y, preds)
    plot_confusion_matrix(cnf_matrix, classes=['Normal', 'Abnormal'], title='Confusion matrix', car=car, attack=attack)


def train2test(car, attack):
    """" training from a file, then testing """
    training_dataset = pd.DataFrame(training_packet)
    training_dataset['timestamp'] = np.log(training_dataset['timestamp'] + 1)
    training_dataset['datalen'] = np.log(training_dataset['datalen'] + 1)
    training_dataset['data'] = np.log(training_dataset['data'] + 1)
    training_dataset['before_canid'] = np.log(training_dataset['before_canid'] + 1)
    training_dataset['after_canid'] = np.log(training_dataset['after_canid'] + 1)

    training_normal = training_dataset[training_dataset['flag'] == 0]
    training_abnormal = training_dataset[training_dataset['flag'] == 1]

    testing_dataset = pd.DataFrame(testing_packet)
    testing_dataset['timestamp'] = np.log(testing_dataset['timestamp'] + 1)
    testing_dataset['datalen'] = np.log(testing_dataset['datalen'] + 1)
    testing_dataset['data'] = np.log(testing_dataset['data'] + 1)
    testing_dataset['before_canid'] = np.log(testing_dataset['before_canid'] + 1)
    testing_dataset['after_canid'] = np.log(testing_dataset['after_canid'] + 1)

    testing_normal = testing_dataset[testing_dataset['flag'] == 0]
    testing_abnormal = testing_dataset[testing_dataset['flag'] == 1]

    train = training_normal
    valid = training_normal.append(training_abnormal)
    test = testing_normal.append(testing_abnormal)

    ## statistical_tec(train, valid, test, car, attack)
    ## linSVM_tec(train, valid, test, car, attack)
    ## rbfSVM_tec(train, valid, test, car, attack)
    ## gaussian_nb(train, valid, test, car, attack)

    # decisiontree_tec(train, valid, test, car, attack)
    # randomforest_tec(train, valid, test, car, attack)
    # kNN_tec(train, valid, test, car, attack)
    voting_classifier(train, valid, test, car, attack)


if __name__=='__main__':
    car_type = ["Sonata", "Soul"]
    attack_type = ["attack"]

    for car in car_type:
        for attack in attack_type:
            clear_dict(training_packet)
            clear_dict(testing_packet)
            print("[+] Start testing " + car + " " + attack)
            read_csv_test(car, attack)            
            read_csv_train(car, attack)
            train2test(car, attack)
            print()
            plt.show()
