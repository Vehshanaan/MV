import os 
import cv2
import json
from tqdm import tqdm


def read_mapping_json(json_path):
    with open(json_path) as f:
        ground_truth_dict = json.load(f)
    return ground_truth_dict


if __name__ == "__main__":
    result_dir = r"E:\appledataset\test_data\detection\yolov5_4"
    json_path = r"E:\test_data\test_data\detection\count_seg_ground_truth.json"
    save_path = result_dir + '_withGT'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    ground_truth_dict = read_mapping_json(json_path)
    for img_name, GT_num in tqdm(ground_truth_dict.items()):
        img_path = os.path.join(result_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.putText(img, "GT:{}".format(GT_num), (50,200), cv2.FONT_HERSHEY_DUPLEX, 2,(0,0,255),8)
        new_img_path = os.path.join(save_path,img_name)
        cv2.imwrite(new_img_path,img)
        