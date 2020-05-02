import os
import csv
from Utils import Message, CURRENT_FOLDER, plot_confusion_matrix

import pandas as pd # for data analytics
import numpy as np # for numerical computation
from matplotlib import pyplot as plt, style # for ploting
from sklearn.metrics import accuracy_score, fbeta_score, precision_score, recall_score, confusion_matrix # for evaluation
from scipy.stats import multivariate_normal
from sklearn.tree import DecisionTreeClassifier, plot_tree

style.use('ggplot')
np.random.seed(42) 

training_packet = {
    'timestamp': [],
    'canid': [],
    'datalen': [],
    'data': [],
    'flag': []
}

# maintesting.py assume that we know the answer
testing_packet = {
    'timestamp': [],
    'canid': [],
    'datalen': [],
    'data': [],
    'flag': []
}

'''
training_packet = {
    'timestamp': [],
    'canid': [],
    'datalen': [],
    'data1': [],
    'data2': [],
    'data3': [],
    'data4': [],
    'data5': [],
    'data6': [],
    'data7': [],
    'data8': [],
    'flag': []
}
testing_packet = {
    'timestamp': [],
    'canid': [],
    'datalen': [],
    'data1': [],
    'data2': [],
    'data3': [],
    'data4': [],
    'data5': [],
    'data6': [],
    'data7': [],
    'data8': [],
    'flag': []
}
'''

# read csv file by path, filename
# code for save data will be add later
def read_file(path, filename, packet):
    with open(os.path.join(path, filename), 'r') as fp:
        reader = csv.reader(fp)
        cnt = 0
        before_timestamp = 0

        for row in reader:

            if len(row) == 0:
                print(" [-] debugging... line " + str(cnt) + " in  " + filename + " is empty")

            elif row[-1] == 'R' or row[-1] == 'T':

                if cnt == 0:
                    before_timestamp = float(row[0])

                cnt += 1
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

                '''
                data = []
                for i in range(8-datalen):
                    data.append(0)
                for idx in range(datalen):
                    data.append(int(msg[idx], 16))
                '''

                timestamp = (float(row[0]) - before_timestamp) * 1000000
                canid = int(row[1], 16)

                if row[-1] == 'R':
                    flag = 0
                elif row[-1] == 'T':
                    flag = 1
                else:
                    print(row[-1])

                packet['timestamp'].append(timestamp)
                packet['canid'].append(canid)
                packet['datalen'].append(datalen)
                packet['data'].append(data)
                '''
                packet['data1'].append(data[0])
                packet['data2'].append(data[1])
                packet['data3'].append(data[2])
                packet['data4'].append(data[3])
                packet['data5'].append(data[4])
                packet['data6'].append(data[5])
                packet['data7'].append(data[6])
                packet['data8'].append(data[7])
                '''
                packet['flag'].append(flag)

                before_timestamp = float(row[0])

            else:
                exit("csv read error")


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

    tresholds = np.linspace(-20, -5, 1000)
    scores = []
    for treshold in tresholds:
        y_hat = (model.logpdf(valid.drop('flag', axis=1).values) < treshold).astype(int)
        scores.append([recall_score(y_pred=y_hat, y_true=valid['flag'].values),
                       precision_score(y_pred=y_hat, y_true=valid['flag'].values),
                       fbeta_score(y_pred=y_hat, y_true=valid['flag'].values, beta=1)])
    scores = np.array(scores)

    plt.plot(tresholds, scores[:, 0], label='$Recall$')
    plt.plot(tresholds, scores[:, 1], label='$Precision$')
    plt.plot(tresholds, scores[:, 2], label='$F_1$')
    plt.ylabel('Score')
    plt.xlabel('Threshold')
    plt.legend(loc='best')
    # plt.show()

    plt.title(car + " " + attack + " f1 score")
    plt.savefig(os.path.join(CURRENT_FOLDER, "output", car, attack + "_f1_score.png"))

    final_tresh = tresholds[scores[:, 2].argmax()]
    y_hat_test = (model.logpdf(test.drop('flag', axis=1).values) < final_tresh).astype(int)

    print('Final threshold: %.3f' % final_tresh)
    print('Test Recall Score: %.3f' % recall_score(y_pred=y_hat_test, y_true=test['flag'].values))
    print('Test Precision Score: %.3f' % precision_score(y_pred=y_hat_test, y_true=test['flag'].values))
    print('Test F1 Score: %.3f' % fbeta_score(y_pred=y_hat_test, y_true=test['flag'].values, beta=1))

    cnf_matrix = confusion_matrix(test['flag'].values, y_hat_test)
    plot_confusion_matrix(cnf_matrix, classes=['Normal', 'Abnormal'], title='Confusion matrix', car=car, attack=attack)


def decisiontree_tec(train, valid, test, car, attack):
    train_X = valid.drop(['flag'], axis=1)
    train_Y = valid['flag']
    test_X = test.drop(['flag'], axis=1)
    test_Y = test['flag']

    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(train_X, train_Y)

    preds = clf.predict(test_X)

    print('Accuracy: {:.4f}%'.format(accuracy_score(test_Y, preds) * 100))
    print('Recall: {:.4f}%'.format(recall_score(test_Y, preds) * 100))
    print('Precision: {:.4f}%'.format(precision_score(test_Y, preds) * 100))
    print('F1 score: {:.4f}%'.format(fbeta_score(test_Y, preds, beta=1) * 100))

    fig, ax = plt.subplots(figsize=(20, 20))
    plot_tree(clf, ax=ax, filled=True)
    plt.show()

    cnf_matrix = confusion_matrix(test_Y, preds)
    plot_confusion_matrix(cnf_matrix, classes=['Normal', 'Abnormal'], title='Confusion matrix', car=car, attack=attack)


def train2test(car, attack):
    """" training from a file, then testing """
    training_dataset = pd.DataFrame(training_packet)
    # training_dataset['timestamp'] = np.log(training_dataset['timestamp'] + 1)
    # training_dataset['canid'] = np.log(training_dataset['canid'] + 1)
    # training_dataset['data'] = np.log(training_dataset['data'] + 1)

    training_normal = training_dataset[training_dataset['flag'] == 0]
    training_abnormal = training_dataset[training_dataset['flag'] == 1]

    testing_dataset = pd.DataFrame(testing_packet)
    # testing_dataset['timestamp'] = np.log(testing_dataset['timestamp'] + 1)
    # testing_dataset['canid'] = np.log(testing_dataset['canid'] + 1)
    # testing_dataset['data'] = np.log(testing_dataset['data'] + 1)

    testing_normal = testing_dataset[testing_dataset['flag'] == 0]
    testing_abnormal = testing_dataset[testing_dataset['flag'] == 1]

    train = training_normal
    valid = training_normal.append(training_abnormal)
    test = testing_normal.append(testing_abnormal)

    # statistical_tec(train, valid, test, car, attack)
    decisiontree_tec(train, valid, test, car, attack)


if __name__=='__main__':
    car_type = ["Sonata", "Soul"]
    attack_type = ["Fuzzy", "Malfunction"]

    for car in car_type:
        for attack in attack_type:
            clear_dict(training_packet)
            clear_dict(testing_packet)

            print("[+] Start testing " + car + " " + attack)
            read_csv_test(car, attack)            
            read_csv_train(car, attack)
            train2test(car, attack)
            print()