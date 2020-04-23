import os
import csv
import sys
import Utils

import pandas as pd # for data analytics
import numpy as np # for numerical computation
from matplotlib import pyplot as plt, style # for ploting
import seaborn as sns # for ploting
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, precision_score, recall_score, confusion_matrix # for evaluation
from scipy.stats import multivariate_normal
import itertools
import missingno as msno

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

# read csv file by path, filename
# code for save data will be add later
def read_file(path, filename, packet):
    with open(os.path.join(path, filename), 'r') as fp:
        reader = csv.reader(fp)
        cnt = 0
        before_timestamp = 0

        for row in reader:
            if row[-1] == 'R' or row[-1] == 'T':
                if cnt == 0:
                    before_timestamp = float(row[0])
                cnt += 1
                datalen = int(row[2])
                data = 0
                # 2019 이용 데이터셋을 제외하고 message packet을 ,로 구분하지 않은 경우가 존재하여 예외처리함 - 동관
                # 후에 예외처리가 더 필요할 수 있음
                msg=[]
                if row[3].count(" "):
                    msg=row[3].split(" ")
                else:
                    msg=row[3:-1]
                for idx in range(datalen):
                    data *= 0x100
                    data += int(msg[idx], 16)

                # timestamp = float(row[0])
                timestamp = int(float(row[0]) - before_timestamp) * 1000000
                # 나중에 log 이용 시 차이가 너무 작지 않도록 조정
                canid = int(row[1], 16)
                if row[-1] == 'R':
                    flag = 0
                elif row[-1] == 'T':
                    flag = 1
                else:
                    print(row[-1])

                # timestamp, CANID, datalen, data, flag must be saved

                packet['timestamp'].append(timestamp)
                packet['canid'].append(canid)
                packet['datalen'].append(datalen)
                packet['data'].append(data)
                packet['flag'].append(flag)

                before_timestamp = float(row[0])

            else:
                exit("csv read error")

        # print(cnt)
        # return packet

def read_csv_train(car_type, attack_type):
    """ params: car_type, attack_type """
    path = os.path.join(Utils.CURRENT_FOLDER, "study_input", car_type)
    file_list = os.listdir(path)
    for file in file_list:
        if attack_type in file:
            read_file(path, file, training_packet)
            break

def read_csv_test(car_type, attack_type):
    """ params: car_type, attack_type """    
    path = os.path.join(Utils.CURRENT_FOLDER, "test_input", car_type)
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

def train2test():
    """" training from a file, then testing """
    training_dataset = pd.DataFrame(training_packet)
    training_dataset['timestamp'] = np.log(training_dataset['timestamp'] + 1)
    training_dataset['canid'] = np.log(training_dataset['canid'] + 1)
    training_dataset['data'] = np.log(training_dataset['data'] + 1)

    training_normal = training_dataset[training_dataset['flag'] == 0]
    training_abnormal = training_dataset[training_dataset['flag'] == 1]

    testing_dataset = pd.DataFrame(testing_packet)
    testing_dataset['timestamp'] = np.log(testing_dataset['timestamp'] + 1)
    testing_dataset['canid'] = np.log(testing_dataset['canid'] + 1)
    testing_dataset['data'] = np.log(testing_dataset['data'] + 1)

    testing_normal = testing_dataset[testing_dataset['flag'] == 0]
    testing_abnormal = testing_dataset[testing_dataset['flag'] == 1]

    """train, _, _, _ = train_test_split(training_normal, training_normal, test_size=.2, random_state=42)
    normal_valid, _, _, _ = train_test_split(training_normal, training_normal, test_size=.5, random_state=42)
    abnormal_valid, _, _, _ = train_test_split(training_abnormal, training_abnormal, test_size=.5, random_state=42)"""

    train = training_normal
    valid = training_normal.append(training_abnormal)
    test = testing_normal.append(testing_abnormal)

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
    """print(scores[:, 2].max(), scores[:, 2].argmax(), tresholds[scores[:, 2].argmax()])

    plt.plot(tresholds, scores[:, 0], label='$Recall$')
    plt.plot(tresholds, scores[:, 1], label='$Precision$')
    plt.plot(tresholds, scores[:, 2], label='$F_1$')
    plt.ylabel('Score')
    plt.xlabel('Threshold')
    plt.legend(loc='best')
    plt.show()"""

    final_tresh = tresholds[scores[:, 2].argmax()]
    y_hat_test = (model.logpdf(test.drop('flag', axis=1).values) < final_tresh).astype(int)

    print('Final threshold: %.3f' % final_tresh)
    print('Test Recall Score: %.3f' % recall_score(y_pred=y_hat_test, y_true=test['flag'].values))
    print('Test Precision Score: %.3f' % precision_score(y_pred=y_hat_test, y_true=test['flag'].values))
    print('Test F1 Score: %.3f' % fbeta_score(y_pred=y_hat_test, y_true=test['flag'].values, beta=1))

    #cnf_matrix = confusion_matrix(test['flag'].values, y_hat_test)
    #plot_confusion_matrix(cnf_matrix, classes=['Normal', 'Abnormal'], title='Confusion matrix')

if __name__=='__main__':
    car_type = ["Sonata"]
    attack_type = ["Fuzzy"]
    # 실제 구현으로는 아래 반복문 안에서 실행시키면 될 것

    for car in car_type:
        for attack in attack_type:
            # 초기화 코드 추가 - 동관
            clear_dict(training_packet)
            clear_dict(testing_packet)            
            read_csv_test(car, attack)            
            read_csv_train(car, attack)
            train2test()