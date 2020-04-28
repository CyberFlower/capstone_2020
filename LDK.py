from Utils import Message

if __name__=='__main__':
    car_type = ["Sonata","Soul"]
    attack_type = ["Fuzzy","Malfunction"]

    for car in car_type:
        for attack in attack_type:
            print("[+] Start testing "+car+" "+attack)            
            msg=Message("study_input",car,attack)
            msg.read_file()
            msg.study_and_test()
            print()