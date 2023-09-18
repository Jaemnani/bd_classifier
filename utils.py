import os

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def set_save_path(path):
    i = 0
    set_path = path
    while(os.path.exists(set_path)):
        set_path = path + "_%.2d"%(i)
        i+=1
        print("process... ", set_path)
    print("Set Save Path Result : ", set_path)
    make_dir(set_path)
    return set_path