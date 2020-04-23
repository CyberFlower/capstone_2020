import os
import csv
import sys
import Utils

# 추가
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


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Copyed from a kernel by joparga3 https://www.kaggle.com/joparga3/kernels
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


packet = {
    'timestamp': [],
    'canid': [],
    'datalen': [],
    'data': [],
    'flag': []
}
# 여기까지


def exit(err):
    print(err)
    sys.exit()


# read csv file by path, filename
# code for save data will be add later
def read_file(path, filename):
    global packet
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
                for idx in range(3, 3 + datalen):
                    data *= 0x100
                    data += int(row[idx], 16)

                # timestamp = float(row[0])
                timestamp = int(float(row[0]) - before_timestamp) * 1000000
                    # 나중에 log 이용 시 차이가 너무 작지 않도록 조정
                canid = int(row[1], 16)
                if row[3 + datalen] == 'R':
                    flag = 0
                elif row[3 + datalen] == 'T':
                    flag = 1
                else:
                    print(row[3 + datalen])

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


# Read csv file by keyword (Car type, Attack type) from study_input
def read_csv_kw(car_type, attack_type):
    path = os.path.join(Utils.CURRENT_FOLDER, "study_input", car_type)
    file_list = os.listdir(path)
    for file in file_list:
        if attack_type in file:
            read_file(path, file)
            break

# clear items from dictionary, but remain keys
# if you want to delete all (include keys), do it just my_dict.clear() 
def clear_dict(my_dict):
    for key in my_dict.keys():
        my_dict[key].clear()

# test using a file
def train_valid_test():
    dataset = pd.DataFrame(packet)
    dataset['timestamp'] = np.log(dataset['timestamp'] + 1)
    dataset['canid'] = np.log(dataset['canid'] + 1)
    dataset['data'] = np.log(dataset['data'] + 1)

    normal = dataset[dataset['flag'] == 0]
    abnormal = dataset[dataset['flag'] == 1]

    train, normal_test, _, _ = train_test_split(normal, normal, test_size=.2, random_state=42)

    normal_valid, normal_test, _, _ = train_test_split(normal_test, normal_test, test_size=.5, random_state=42)
    abnormal_valid, abnormal_test, _, _ = train_test_split(abnormal, abnormal, test_size=.5, random_state=42)

    train = train.reset_index(drop=True)
    valid = normal_valid.append(abnormal_valid).sample(frac=1).reset_index(drop=True)
    test = normal_test.append(abnormal_test).sample(frac=1).reset_index(drop=True)

    # print('Train shape: ', train.shape)
    # print('Proportion os anomaly in training set: %.2f\n' % train['flag'].mean())
    # print('Valid shape: ', valid.shape)
    # print('Proportion os anomaly in validation set: %.2f\n' % valid['flag'].mean())
    # print('Test shape:, ', test.shape)
    # print('Proportion os anomaly in test set: %.2f\n' % test['flag'].mean())

    mu = train.drop('flag', axis=1).mean(axis=0).values
    sigma = train.drop('flag', axis=1).cov().values
    model = multivariate_normal(cov=sigma, mean=mu, allow_singular=True)

    # print(np.median(model.logpdf(valid[valid['flag'] == 0].drop('flag', axis=1).values)))
    # print(np.median(model.logpdf(valid[valid['flag'] == 1].drop('flag', axis=1).values)))

    tresholds = np.linspace(-20, -5, 1000)
    scores = []
    for treshold in tresholds:
        y_hat = (model.logpdf(valid.drop('flag', axis=1).values) < treshold).astype(int)
        scores.append([recall_score(y_pred=y_hat, y_true=valid['flag'].values),
                       precision_score(y_pred=y_hat, y_true=valid['flag'].values),
                       fbeta_score(y_pred=y_hat, y_true=valid['flag'].values, beta=1)])

    scores = np.array(scores)
    print(scores[:, 2].max(), scores[:, 2].argmax(), tresholds[scores[:, 2].argmax()])

    plt.plot(tresholds, scores[:, 0], label='$Recall$')
    plt.plot(tresholds, scores[:, 1], label='$Precision$')
    plt.plot(tresholds, scores[:, 2], label='$F_1$')
    plt.ylabel('Score')
    plt.xlabel('Threshold')
    plt.legend(loc='best')
    plt.show()

    final_tresh = tresholds[scores[:, 2].argmax()]
    y_hat_test = (model.logpdf(test.drop('flag', axis=1).values) < final_tresh).astype(int)

    print('Final threshold: %.3f' % final_tresh)
    print('Test Recall Score: %.3f' % recall_score(y_pred=y_hat_test, y_true=test['flag'].values))
    print('Test Precision Score: %.3f' % precision_score(y_pred=y_hat_test, y_true=test['flag'].values))
    print('Test F1 Score: %.3f' % fbeta_score(y_pred=y_hat_test, y_true=test['flag'].values, beta=1))

    cnf_matrix = confusion_matrix(test['flag'].values, y_hat_test)
    plot_confusion_matrix(cnf_matrix, classes=['Normal', 'Abnormal'], title='Confusion matrix')

if __name__ == "__main__":
    # car = ["Sonata", "Soul", "Spark"]
    # attack = ["Flooding", "Fuzzy", "Malfunction"]

    car_type = ["Sonata"]
    attack_type = ["Flooding"]
    # 실제 구현으로는 아래 반복문 안에서 실행시키면 될 것

    for car in car_type:
        for attack in attack_type:
            # 초기화 코드 추가 - 동관
            clear_dict(packet)
            read_csv_kw(car, attack)
            train_valid_test()

