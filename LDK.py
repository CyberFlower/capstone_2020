from Utils import Message

if __name__=='__main__':
    car_type = ["Sonata","Soul"]
    attack_type = ["Fuzzy","Malfunction"]
    path_type=["study_input","test_input"]
    
    '''for car in car_type:
        for attack in attack_type:
            for folder in path_type:
                print("[+] Start testing " + car + " " + attack)
                msg = Message(folder,car,attack)
                msg.read_file()
                #msg.study_and_test()
                #msg.scatter_graph()
                #msg.rev_scatter_graph()
                #msg.no_log_scatter_graph()
                #msg.rev_no_log_scatter_graph()
                #msg.scatter_graph_time()
                msg.scatter_graph_id_relate()    
                msg.scatter_graph_id_relate(rev=True)                
                print()'''

    for car in car_type:
        for folder in path_type:
            car1=Message(folder,car,"Fuzzy")
            car2=Message(folder,car,"Malfunction")
            car1.read_file()
            car2.read_file()
            car1.merge_file_by_time(car2)

    
    