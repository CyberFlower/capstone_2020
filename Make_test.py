import Utils
import os
import csv

def make_query(path,filename):
    line=1
    with open(os.path.join(path, filename), 'r') as fp:
        reader = csv.reader(fp)
        with open(os.path.join(path,"query_"+filename),'w') as write_fp:
            pass


# Read csv file by keyword (Car type, Attack type) from test_input
def read_csv_kw(car_type, attack_type):
    path = os.path.join(Utils.CURRENT_FOLDER, "test_input", car_type)
    file_list = os.listdir(path)
    for file in file_list:
        if attack_type in file:
            make_query(path, file)
            break

if __name__ == "__main__":
    pass