import os, xmltodict

class_id = 9
path = 'data/salad/'

train_files=[]
test_files=[]

files = os.listdir(path)
i=0
for f in files:
    if f.endswith('.xml'):
        with open(path+f) as fd:
            doc = xmltodict.parse(fd.read())
            
        width = int(doc['annotation']['size']['width'])
        height = int(doc['annotation']['size']['height'])
        
        xmin = int(doc['annotation']['object']['bndbox']['xmin'])
        ymin = int(doc['annotation']['object']['bndbox']['ymin'])
        xmax = int(doc['annotation']['object']['bndbox']['xmax'])
        ymax = int(doc['annotation']['object']['bndbox']['ymax'])
            
        x = (xmin+xmax)/2/width
        y = (ymin+ymax)/2/height
        object_width = (xmax-xmin)/width
        object_height = (ymax-ymin)/height
            
        with open(path+f[:-3]+'txt', 'w') as fd:
            fd.write(str(class_id)+' '+ str(x)+' '+ str(y)+' '+ str(object_width)+' '+ str(object_height))