import os, csv
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, style
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, precision_score, recall_score, confusion_matrix
from scipy.stats import multivariate_normal
import itertools

style.use('ggplot')
#style.use('fivethirtyeight')
np.random.seed(42) 

CURRENT_FOLDER=os.path.dirname(os.path.abspath(__file__))


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues, car="", attack=""):
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
    # plt.show()
    plt.title(car + " " + attack + " confusion matrix")
    plt.savefig(os.path.join(CURRENT_FOLDER, "output", car,attack + "_confusion_matrix.png"))


class Message:
    def __init__(self, folder, car_type, attack_type):
        """ Datas from a file 
        params: folder, car_type, attack_type
        folder: study_input or test_input (select a folder to your purpose)
        """
        self.packet = {
            'message': [],
            'timestamp': [],
            'before_canid': [],
            'canid': [],
            'after_canid': [],
            'datalen': [],
            'data': [],
            'flag': []
        }        
        self.folder=folder
        self.path = os.path.join(CURRENT_FOLDER, self.folder, car_type)
        self.car=car_type
        self.attack=attack_type
        self.filename=""
        self.ln=0
        file_list = os.listdir(self.path)
        for file in file_list:
            if attack_type in file:
                self.filename=file
                break
        
    def read_file(self):
        with open(os.path.join(self.path, self.filename), 'r') as fp:
            reader = csv.reader(fp)
            cnt = 0
            before_timestamp = 0
            before_id = 0

            for row in reader:
                if row[-1] == 'R' or row[-1] == 'T':
                    self.packet['message'].append(row)

                    if cnt == 0:
                        before_timestamp = float(row[0])

                    datalen = int(row[2])
                    data = 0

                    msg = []
                    if row[3].count(" "):
                        msg = row[3].split(" ")
                    else:
                        msg=row[3:-1]
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

                    self.packet['timestamp'].append(timestamp)
                    self.packet['before_canid'].append(before_canid)
                    # self.packet['after_canid'].append(after_canid)
                    self.packet['canid'].append(canid)
                    self.packet['datalen'].append(datalen)
                    self.packet['data'].append(data)
                    self.packet['flag'].append(flag)
                    self.ln=self.ln+1

                    before_timestamp = float(row[0])

                    if cnt != 0:
                        self.packet['after_canid'].append(canid)  # 두 번째 줄부터 after_canid 를 추가하면 자동으로 순서가 맞춰짐

                    cnt += 1

                else:
                    exit("csv read error")

            self.packet['after_canid'].append(0)  # 마지막 메시지의 after_canid 는 0

    def get_packet(self):
        return self.packet

    def file_copy(self):
        with open(os.path.join(self.path, "copy_"+self.filename), 'w') as fp:
            writer=csv.writer(fp,delimiter=' ')
            for i in range(self.ln):
                writer.writerow([self.packet['timestamp'][i],self.packet['canid'][i],self.packet['data'][i],self.packet['flag'][i]])

    def scatter_graph(self):
        dataset = pd.DataFrame(self.packet)
        #dataset['timestamp'] = np.log(dataset['timestamp'] + 1)
        #dataset['datalen'] = np.log(dataset['datalen'] + 1)
        dataset['data'] = np.log(dataset['data'] + 1)
        dataset['canid']=self.packet['canid']
        #dataset['threat']=self.packet['flag']

        normal = dataset[dataset['flag'] == 0]
        abnormal = dataset[dataset['flag'] == 1]
        
        fig, ax = plt.subplots()
        for color in ["red", "blue"]:
            if color=="blue":
                ax.scatter(normal['canid'],normal['data'],c=color,label='normal')
            else:
                ax.scatter(abnormal['canid'],abnormal['data'],c=color,label='abnormal')
        ax.legend()
        ax.grid(True)

        plt.title(self.car+" "+self.attack+" "+"Scatter")
        plt.xlabel("CAN ID")
        plt.ylabel("Logarithm of Attack Message")
        if not os.path.exists(os.path.join(self.path,"img")):
            os.makedirs(os.path.join(self.path,"img"))
        plt.savefig(os.path.join(self.path,"img","scatter_"+self.car+"_"+self.attack+".png"))

    def rev_scatter_graph(self):
        dataset = pd.DataFrame(self.packet)
        #dataset['timestamp'] = np.log(dataset['timestamp'] + 1)
        #dataset['datalen'] = np.log(dataset['datalen'] + 1)
        dataset['data'] = np.log(dataset['data'] + 1)
        dataset['canid']=self.packet['canid']
        #dataset['threat']=self.packet['flag']

        normal = dataset[dataset['flag'] == 0]
        abnormal = dataset[dataset['flag'] == 1]
        
        fig, ax = plt.subplots()
        for color in ["red", "blue"]:
            if color=="red":
                ax.scatter(normal['canid'],normal['data'],c=color,label='normal')
            else:
                ax.scatter(abnormal['canid'],abnormal['data'],c=color,label='abnormal')
        ax.legend()
        ax.grid(True)

        plt.title(self.car+" "+self.attack+" "+"Scatter")
        plt.xlabel("CAN ID")
        plt.ylabel("Logarithm of Attack Message")
        if not os.path.exists(os.path.join(self.path,"img")):
            os.makedirs(os.path.join(self.path,"img"))        
        plt.savefig(os.path.join(self.path,"img","rev_scatter_"+self.car+"_"+self.attack+".png"))

    def scatter_graph_time(self, rev=False):
        dataset = pd.DataFrame(self.packet)
        dataset['timestamp'] = np.log(dataset['timestamp'] + 1)
        #dataset['datalen'] = np.log(dataset['datalen'] + 1)
        #dataset['data'] = np.log(dataset['data'] + 1)
        dataset['canid']=self.packet['canid']
        #dataset['threat']=self.packet['flag']

        normal = dataset[dataset['flag'] == 0]
        abnormal = dataset[dataset['flag'] == 1]
        
        fig, ax = plt.subplots()
        for color in ["red", "blue"]:
            if (rev and color=="blue") or (not rev and color=="red"):
                ax.scatter(normal['canid'],normal['timestamp'],c=color,label='normal')
            else:
                ax.scatter(abnormal['canid'],abnormal['timestamp'],c=color,label='abnormal')
        ax.legend()
        ax.grid(True)

        plt.title(self.car+" "+self.attack+" "+"Scatter")
        plt.xlabel("CAN ID")
        plt.ylabel("Logarithm of Timestamp")
        if not os.path.exists(os.path.join(self.path,"img")):
            os.makedirs(os.path.join(self.path,"img"))
        file_title=""
        if rev:
            file_title="time_scatter_"+self.car+"_"+self.attack+".png"
        else:
            file_title="rev_time_scatter_"+self.car+"_"+self.attack+".png"

        plt.savefig(os.path.join(self.path,"img",file_title))

    def scatter_graph_id_relate(self, rev=False):
        dataset = pd.DataFrame(self.packet)
        dataset['data'] = np.log(dataset['data'] + 1)
        dataset['id_relate'] = np.log(dataset['id_relate'] + 1)

        normal = dataset[dataset['flag'] == 0]
        abnormal = dataset[dataset['flag'] == 1]
        
        fig, ax = plt.subplots()
        for color in ["red", "blue"]:
            if (rev and color=="blue") or (not rev and color=="red"):
                ax.scatter(normal['canid'],normal['id_relate'],c=color,label='normal')
            else:
                ax.scatter(abnormal['canid'],abnormal['id_relate'],c=color,label='abnormal')
        ax.legend()
        ax.grid(True)

        plt.title(self.car+" "+self.attack+" "+"Scatter")
        plt.xlabel("CAN ID")
        plt.ylabel("Logarithm of ID relation")
        if not os.path.exists(os.path.join(self.path,"img")):
            os.makedirs(os.path.join(self.path,"img"))
        file_title=""
        if rev:
            file_title="id_relate_scatter_"+self.car+"_"+self.attack+".png"
        else:
            file_title="rev_id_relate_scatter_"+self.car+"_"+self.attack+".png"

        plt.show()
        # plt.savefig(os.path.join(self.path,"img",file_title))

    def no_log_scatter_graph(self):
        dataset = pd.DataFrame(self.packet)
        #dataset['timestamp'] = np.log(dataset['timestamp'] + 1)
        #dataset['datalen'] = np.log(dataset['datalen'] + 1)
        dataset['data'] = dataset['data']
        dataset['canid']=self.packet['canid']
        #dataset['threat']=self.packet['flag']

        normal = dataset[dataset['flag'] == 0]
        abnormal = dataset[dataset['flag'] == 1]
        
        fig, ax = plt.subplots()
        for color in ["red", "blue"]:
            if color=="blue":
                ax.scatter(normal['canid'],normal['data'],c=color,label='normal')
            else:
                ax.scatter(abnormal['canid'],abnormal['data'],c=color,label='abnormal')
        ax.legend()
        ax.grid(True)

        plt.title(self.car+" "+self.attack+" "+"Scatter")
        plt.xlabel("CAN ID")
        plt.ylabel("Attack Message")
        if not os.path.exists(os.path.join(self.path,"img")):
            os.makedirs(os.path.join(self.path,"img"))
        plt.savefig(os.path.join(self.path,"img","no_log_scatter_"+self.car+"_"+self.attack+".png"))

    def rev_no_log_scatter_graph(self):
        dataset = pd.DataFrame(self.packet)
        #dataset['timestamp'] = np.log(dataset['timestamp'] + 1)
        #dataset['datalen'] = np.log(dataset['datalen'] + 1)
        dataset['data'] = dataset['data']
        dataset['canid']=self.packet['canid']
        #dataset['threat']=self.packet['flag']

        normal = dataset[dataset['flag'] == 0]
        abnormal = dataset[dataset['flag'] == 1]
        
        fig, ax = plt.subplots()
        for color in ["red", "blue"]:
            if color=="red":
                ax.scatter(normal['canid'],normal['data'],c=color,label='normal')
            else:
                ax.scatter(abnormal['canid'],abnormal['data'],c=color,label='abnormal')
        ax.legend()
        ax.grid(True)

        plt.title(self.car+" "+self.attack+" "+"Scatter")
        plt.xlabel("CAN ID")
        plt.ylabel("Attack Message")
        if not os.path.exists(os.path.join(self.path,"img")):
            os.makedirs(os.path.join(self.path,"img"))
        plt.savefig(os.path.join(self.path,"img","rev_no_log_scatter_"+self.car+"_"+self.attack+".png"))

    def study_and_test(self):
        """ this function study from a file, and test this file
        """
        dataset = pd.DataFrame(self.packet)
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
        #print(scores[:, 2].max(), scores[:, 2].argmax(), tresholds[scores[:, 2].argmax()])

        plt.plot(tresholds, scores[:, 0], label='$Recall$')
        plt.plot(tresholds, scores[:, 1], label='$Precision$')
        plt.plot(tresholds, scores[:, 2], label='$F_1$')
        plt.ylabel('Score')
        plt.xlabel('Threshold')
        plt.legend(loc='best')
        #plt.show()
        plt.title(self.car+" "+self.attack+" f1 score")
        plt.savefig(os.path.join(CURRENT_FOLDER,"output",self.car,self.attack+"_f1_score.png"))        

        final_tresh = tresholds[scores[:, 2].argmax()]
        y_hat_test = (model.logpdf(test.drop('flag', axis=1).values) < final_tresh).astype(int)

        print('Final threshold: %.3f' % final_tresh)
        print('Test Recall Score: %.3f' % recall_score(y_pred=y_hat_test, y_true=test['flag'].values))
        print('Test Precision Score: %.3f' % precision_score(y_pred=y_hat_test, y_true=test['flag'].values))
        print('Test F1 Score: %.3f' % fbeta_score(y_pred=y_hat_test, y_true=test['flag'].values, beta=1))

        cnf_matrix = confusion_matrix(test['flag'].values, y_hat_test)
        plot_confusion_matrix(cnf_matrix, classes=['Normal', 'Abnormal'], title='Confusion matrix',car=self.car,attack=self.attack)

    # sorted list A + sorted list B -> sorted list C 
    def merge_file_by_time(self, other):
        left=self.packet['message']
        right=other.packet['message']
        l=0; r=0
        res=[]
        while l<len(left) and r<len(right):
            if left[l][0]==right[r][0]:
                print("[-] Time equal error!")
            if float(left[l][0])<float(right[r][0]):
                res.append(left[l])
                l+=1
            else:
                res.append(right[r])
                r+=1
        while l<len(left):
            res.append(left[l])
            l+=1
        while r<len(right):
            res.append(right[r])
            r+=1

        with open(os.path.join(self.path, self.car+"_attack.csv"), 'w') as fp:
            writer=csv.writer(fp,delimiter=',')
            for i in range(len(res)):
                writer.writerow(res[i])
