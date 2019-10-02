import numpy as np
import os
import tensorflow as tf
import cv2
import imageio
import datetime
import time

from skimage import color
from object_detection.utils import label_map_util
#from joblib import Parallel, delayed
from skimage import img_as_ubyte
from object_detection.utils import visualization_utils as vis_util

PATH_TO_CKPT = 'training_rcnn_resnet50/frozen_model/frozen_inference_graph.pb'
PATH_TO_LABELS = 'label_map.pbtxt'
NUM_CLASSES = 9


def cut_and_resize_object_from_image(image_np, box, width, heigth):
    part = image_np[int(box[0]*width):int(box[2]*width)
        , int(box[1]*heigth):int(box[3]*heigth), :]
    part = color.gray2rgb(color.rgb2gray(part))
    #part = (part-part.min()) / (part.max()-part(min)) #normalizes data in range 0 - 255
    part = 255 * part
    part = part.astype(np.uint8)
    return part


def get_objects_from_image(image_np_expanded, sess):
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')      
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')

    (boxes, scores, classes) = sess.run(
        [boxes, scores, classes],
        feed_dict={image_tensor: image_np_expanded})
    return boxes, scores, classes


def get_good_boxes(boxes, scores, classes):
    good_boxes=[]
    good_scores=[]
    good_classes=[]
    for i, box in enumerate(np.squeeze(boxes)):
        if(np.squeeze(scores)[i] > 0.6 and
            box[2]>box[0] and
            box[3]>box[1]):
            good_boxes.append(box)
            good_scores.append(np.squeeze(scores)[i])
            good_classes.append(np.squeeze(classes)[i])
    return good_boxes, good_scores, good_classes

def get_classnames_list(good_classes, category_index):
    ret = []
    for g in good_classes:
        ret.append(category_index[int(g)]['name'])
    return ret


def process_box(image_np, box, width, heigth, name):
    if(len(box)==4):
        part = cut_and_resize_object_from_image(image_np, box, width, heigth)
        imageio.imsave(name, part)

def get_sess():
    with detection_graph.as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(graph=detection_graph, config=config)
    return sess

def get_goods(image_np, person_sess): 
    image_np_expanded = np.expand_dims(image_np, axis=0)
    boxes, scores, classes = get_objects_from_image(image_np_expanded, person_sess)
    good_boxes = get_good_boxes(boxes, scores, classes)
    (width, heigth)=(image_np.shape[0], image_np.shape[1])
    persons = []
    for i, box in enumerate(good_boxes):
        persons.append( cut_and_resize_object_from_image(image_np, box, width, heigth))
    return persons

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)

category_index = label_map_util.create_category_index(categories)

def recognize_goods_in_dir(source_dir, cat_dir):
#Получаем список файлов в переменную files 
    worked_files=[]
    files_for_work=[]
    sess=get_sess()
    files = os.listdir(source_dir)
    files_for_work= list(set(files) - set(worked_files))
    
    rec_classes={}
    rec_classes['angle']=0
    rec_classes['line']=0
    rec_classes['z']=0
    rec_classes['dot']=0
    rec_classes['square']=0
    
    rec_classes['unrec']=0
    
    for f in sorted(files_for_work):
        if f.endswith('jpg'):
            print(f, datetime.datetime.now())
            filename = f.split('.')[0]
            image_np_raw=imageio.imread(os.path.join(source_dir,f))
            image_np = img_as_ubyte(image_np_raw)
            (width, heigth)=(image_np.shape[0], image_np.shape[1])
            image_np_expanded = np.expand_dims(image_np, axis=0)
            boxes, scores, classes = get_objects_from_image(image_np_expanded, sess)
            
            good_boxes, good_scores, good_classes = get_good_boxes(boxes, scores, classes)
            for i, box in enumerate(good_boxes):
                rec_classes[category_index[int(good_classes[i])]['name']]+=1
                process_box(image_np, good_boxes[i], width, heigth
                            , os.path.join(cat_dir+'/'+category_index[int(good_classes[i])]['name'],filename+'_obj_'+str(i)+'_score_'+str(good_scores[i])+'.jpg'))
            if len(good_boxes)==0:
                imageio.imsave(os.path.join(cat_dir+'/'+'unrec',f), image_np)
                rec_classes['unrec']+=1
            # накладываем на массив bounding boxes
            #if len(good_boxes)>0:
            #    names = [get_classnames_list(good_classes, category_index)]
            #    vis_util.draw_bounding_boxes_on_image_array(image_np, np.array(good_boxes), display_str_list_list=names)
            #cv2.imshow("Input", image_np)
            #time.sleep(0.5)
            if cv2.waitKey(27) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
    worked_files+=files_for_work
    cv2.destroyAllWindows()
    print(rec_classes)
    del sess

    


#Каталог из которого будем брать изображения 
source_dir = 'data/test_frame/z' 
cat_dir = 'data/recognized/z'
recognize_goods_in_dir(source_dir, cat_dir)