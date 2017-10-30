from TestPrint import TestPrint

printparams = {
    'data_root': './data/images/', 
    'data_result_list': './data/testprint2.txt',
    'test_num': 10000,
    'model': net,
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean
    }

p = TestAndPrint(**printparams)
p.PrintToFile()