import os
import glob
import random
from shutil import copy

def main():
    
    train_folders = ['angle','dot','line','square','z','color_wok','craft_wok','cup','salad','test_color_wok','test_craft_wok','test_cup','test_salad']
    jpg_list = []
    for f in train_folders:
        for jpg_file in glob.glob(os.path.join(os.getcwd(),'data', f) + '/*.JPG'):
            copy( jpg_file, os.path.join(os.getcwd(),'data_darknet', 'images', '001'))
          
        for txt_file in glob.glob(os.path.join(os.getcwd(),'data', f) + '/*.txt'):
            jpg_list.append(os.path.join(os.getcwd(),'data_darknet', 'images', '001', os.path.basename(txt_file)[:-3]+'jpg'))
            copy( txt_file, os.path.join(os.getcwd(),'data_darknet', 'labels', '001'))
            

    print('Successfully copied files.')
    
    for f in jpg_list:
        if random.random()<0.8:            
            with open(os.path.join(os.getcwd(),'data_darknet', 'train_files.txt'), 'a') as fd:
                fd.write(f+'\n')
        else:            
            with open(os.path.join(os.getcwd(),'data_darknet', 'test_files.txt'), 'a') as fd:
                fd.write(f+'\n')


    print('Successfully train and test files made.')

if __name__ == '__main__':
    main()