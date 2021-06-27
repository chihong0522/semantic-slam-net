import numpy as np
import xml.dom.minidom
import xml.etree.ElementTree as ET
def main():
    xml = '/home/nvidia/semantic_slam_ws/src/semantic_cloud/include/multitask_refinenet/nyu_color.xml'
    npy ='/home/nvidia/semantic_slam_ws/src/semantic_cloud/include/multitask_refinenet/cmap_nyud_bgr8.npy'
    # scenenn_xml = '/home/nvidia/SceneNN-dataset/scenenn/014/014.xml'
    # sceneNN_color_map = parse_sceneNN_color(scenenn_xml)

    np_arr = np.load(npy)
    color_list = [] 
    
    tree = ET.parse(xml)
    root = tree.getroot()
    for child in root:
        cmap = child.attrib
        # bgr8 = [int(x) for x in cmap['sem_map_color'].split(' ')]
        bgr8 = [int(x) for x in cmap['color'].split(' ')]
        id = int(cmap['id'])
        color_list.insert(id, bgr8)
    np_cmap = np.array(color_list,dtype='uint8')
    np.save(npy, np_cmap)
    print("Save to",npy)

def parse_sceneNN_color(xml_):
    sceneNN_color = {}
    tree = ET.parse(xml_)
    root = tree.getroot()
    for child in root:
        cmap = child.attrib
        rgb8 = [int(x) for x in cmap['color'].split(' ')]
        rgb8.reverse()
        id = int(cmap['id'])
        if cmap['nyu_class']=='prop':
            cmap['nyu_class'] = 'otherprop'
        if len(cmap['nyu_class'])>0:
            sceneNN_color[cmap['nyu_class']] = rgb8

    return sceneNN_color


main()