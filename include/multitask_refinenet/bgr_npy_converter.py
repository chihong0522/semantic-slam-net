import numpy as np
import xml.dom.minidom
import xml.etree.ElementTree as ET
def main():
    xml = '/home/chihung/semantic_map_ws/src/semantic_cloud/include/multitask_refinenet/nyu_color.xml'
    npy ='/home/chihung/semantic_map_ws/src/semantic_cloud/include/multitask_refinenet/cmap_nyud_bgr8.npy'
    
    np_arr = np.load(npy)
    color_list = [] 
    
    tree = ET.parse(xml)
    root = tree.getroot()
    for child in root:
        cmap = child.attrib
        bgr8 = [int(x) for x in cmap['color'].split(' ')]
        id = int(cmap['id'])
        color_list.insert(id, bgr8)
    np_cmap = np.array(color_list,dtype='uint8')
    np.save(npy, np_cmap)
    print("Save to",npy)

main()