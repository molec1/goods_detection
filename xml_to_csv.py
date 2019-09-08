import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path, image_subfolder):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (os.path.join(image_subfolder, root.find('filename').text),
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    return xml_list


def main():
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    
    train_folders = ['data\color_wok','data\craft_wok','data\cup','data\salad']
    xml_list = []
    for f in train_folders:
        image_path = os.path.join(os.getcwd(), f)
        xml_list += (xml_to_csv(image_path, f))
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    xml_df.to_csv('all_goods.csv', index=None)
    print('Successfully converted xml to csv.')


if __name__ == '__main__':
    main()