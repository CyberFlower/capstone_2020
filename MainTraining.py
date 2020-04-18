import os
import csv
import sys
import Utils

def exit(err):
    print(err)
    sys.exit()

def read_file(path, filename):
    with open(os.path.join(path,filename),'r') as fp:
        reader=csv.reader(fp)
        cnt=0
        for row in reader:
            if row[-1]=='R':
                cnt+=1
            elif row[-1]=='T':
                cnt+=1
            else:
                exit("csv read error")
        print(cnt)

# Read csv file by keyword (Car type, Attack type) from study_input
def read_csv_kw(car_type, attack_type):
    path=os.path.join(Utils.CURRENT_FOLDER, "study_input",car_type)
    file_list=os.listdir(path)
    for file in file_list:
        if attack_type in file:
            read_file(path,file)


if __name__ == "__main__":
    read_csv_kw("Sonata","Fuzzy")