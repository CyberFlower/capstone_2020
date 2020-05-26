from Utils import Message

if __name__=='__main__':
    car_type = ["Sonata","Soul"]
    attack_type = ["Fuzzy","Malfunction"]
    path_type=["study_input","test_input"]
    
    for car in car_type:
        for attack in attack_type:
            for folder in path_type:
                print("[+] Start testing "+car+" "+attack)            
                msg=Message(folder,car,attack)
                msg.read_file()
                #msg.study_and_test()
                #msg.scatter_graph()
                #msg.rev_scatter_graph()
                #msg.no_log_scatter_graph()
                #msg.rev_no_log_scatter_graph()
                #msg.scatter_graph_time()
                msg.scatter_graph_id_relate()    
                msg.scatter_graph_id_relate(rev=True)                
                print()

    '''for car in car_type:
        for attack in attack_type:
            print("[+] Start testing "+car+" "+attack)            
            msg=Message("test_input",car,attack)
            msg.read_file()
            packet=msg.get_packet()
            cnt=0; sz=len(packet['flag'])
            malfunction_set=set()
            for i in range(sz):
                if packet['flag'][i]==1:
                    xx=str(packet['data'][i])
                    malfunction_set.add(xx)
            #print(len(malfunction_set))
            for attack_messages in malfunction_set:
                print(attack_messages)'''

    
    