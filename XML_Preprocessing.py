import os
import glob
import shutil
from xml.dom import minidom
import numpy as np


def add_nodes(xml_file_path):
    xml = minidom.parse(xml_file_path)
    x = xml.documentElement
    y = xml.getElementsByTagName('Region')

    for i in range(len(y)):
        newNode = xml.createElement('Attribute')
        y = xml.getElementsByTagName('Region')[i]
        y.insertBefore(newNode, y.childNodes[1])

    with open(xml_file_path,'w') as f:
        xml.writexml(f)
        f.close()


def add_attribute(xml_file_path):
    xml = minidom.parse(xml_file_path)
    # data = xml.documentElement
    nodelist = xml.getElementsByTagName('Region')

    children = []
    tag = []
    for i in range(len(nodelist)):
        child = nodelist[i].childNodes
        children.append(child)

    for ele, child in enumerate(children):
        a_label = nodelist[ele].attributes['NegativeROA'].value
        b_label = nodelist[ele].parentNode.parentNode.attributes['Id'].value
        c_label = int(a_label) + int(b_label)

        # print(child[1])

        if c_label == 1:
            child[1].setAttribute('Value', 'Whole Tumor')
        elif c_label == 2:
            child[1].setAttribute('Value', 'Viable Tumor')
        elif c_label == 3:
            child[1].setAttribute('Value', 'Negative Pen')

        tag.append((child[1].getAttribute('Value')))

    with open(xml_file_path, 'w') as f:
        xml.writexml(f)
        f.close()


def get_all_xml_file(total_file_path,xml_file_path):
    file_path = total_file_path  # 所有文件的地址
    xml_path = xml_file_path  # 所有xml文件的地址
    files = os.listdir(file_path)

    for file in files:
        if file.endswith('.xml'):
            shutil.copy(os.path.join(file_path,file),xml_path)
        elif file.endswith('.svs'):
            shutil.copy(os.path.join(file_path, file), xml_path)
        elif file.endswith('.SVS'):
            shutil.copy(os.path.join(file_path, file), xml_path)

    file_name = next(os.walk(xml_path))[2]
    return  file_name


if __name__ == '__main__':
    file_path = r'C:\XZ\Experiment\Medical\data\SourceData\Image'
    xml_path = r'C:\XZ\Experiment\Medical\data\SourceData\XML'
    file_name = get_all_xml_file(file_path,xml_path)

    for file_ in file_name:
        if file_.endswith('.xml'):
            xml_file_path = os.path.join(xml_path,file_)
            add_nodes(xml_file_path)
            add_attribute(xml_file_path)
        else:
            pass

