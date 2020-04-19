import os
import csv
import sys
import Utils

# 추가
import pandas as pd # for data analytics
import numpy as np # for numerical computation
from matplotlib import pyplot as plt, style # for ploting
import seaborn as sns # for ploting
from sklearn.metrics import fbeta_score, precision_score, recall_score, confusion_matrix # for evaluation
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
    'packet_timestamp': [],
    'packet_canid': [],
    'packet_datalen': [],
    'packet_data': [],
    'packet_flag': []
}
# 여기까지


def exit(err):
    print(err)
    sys.exit()


# read csv file by path, filename
# code for save data will be add later
def read_file(path, filename):
    with open(os.path.join(path, filename), 'r') as fp:
        reader = csv.reader(fp)
        cnt = 0

        for row in reader:
            if row[-1] == 'R' or row[-1] == 'T':
                cnt += 1
                datalen = int(row[2])
                data = 0
                for idx in range(3, 3 + datalen):
                    data *= 0x100
                    data += int(row[idx], 16)

                timestamp = float(row[0])
                canid = int(row[1], 16)
                if row[3 + datalen] == 'R':
                    flag = 0
                elif row[3 + datalen] == 'T':
                    flag = 1
                else:
                    print(row[3 + datalen])

                # timestamp, CANID, datalen, data, flag must be saved

                packet['packet_timestamp'].append(timestamp)
                packet['packet_canid'].append(canid)
                packet['packet_datalen'].append(datalen)
                packet['packet_data'].append(data)
                packet['packet_flag'].append(flag)

            else:
                exit("csv read error")

        print(cnt)
        return packet


# Read csv file by keyword (Car type, Attack type) from study_input
def read_csv_kw(car_type, attack_type):
    path = os.path.join(Utils.CURRENT_FOLDER, "study_input", car_type)
    file_list = os.listdir(path)
    for file in file_list:
        if attack_type in file:
            read_file(path, file)
            break


if __name__ == "__main__":
    # car = ["Sonata", "Soul", "Spark"]
    # attack = ["Flooding", "Fuzzy", "Malfunction"]

    car = ["Sonata"]
    attack = ["Flooding"]
    # 실제 구현으로는 아래 반복문 안에서 실행시키면 될 것

    for xx in car:
        for yy in attack:
            print(xx + " " + yy)
            read_csv_kw(xx, yy)
            # 여기에서 실행하고 반복문 끝무렵 초기화
            # packet.clear()
            # packet = {
            #     'packet_timestamp': [],
            #     'packet_canid': [],
            #     'packet_datalen': [],
            #     'packet_data': [],
            #     'packet_flag': []
            # }
            # 추후 초기화 부분 디버깅 필

    dataset = pd.DataFrame(packet)







    