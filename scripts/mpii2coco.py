from scipy.io import loadmat, savemat
from PIL import Image
import os
import os.path
import numpy as np
import json
import argparse

MPII_ANNOTATIONS = 'mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat'
MPII_IMAGES_FOLDER = 'mpii_human_pose_v1/images'
JOINT_NUM = 16


def check_empty(list, name):
    if list is None:
        return True
    try:
        list[name]
    except ValueError:
        return True

    if len(list[name]) > 0:
        return False
    else:
        return True


def convert_to_coco(dataset_path, train_path, test_path):
    annotation_file = loadmat(os.path.join(dataset_path, MPII_ANNOTATIONS))
    mpii = {n: annotation_file['RELEASE'][n][0, 0]
            for n in annotation_file['RELEASE'].dtype.names}
    annolist = mpii['annolist'][0]
    img_num = len(annolist['image'])
    aid = 0
    coco_train = {'images': [], 'categories': [], 'annotations': []}
    coco_test = {'images': [], 'categories': [], 'annotations': []}

    for img_id in range(img_num):
        image_annolist = annolist[img_id]
        # any person is annotated
        if check_empty(image_annolist, 'annorect') == False:
            # filename
            image_file = os.path.join(dataset_path, MPII_IMAGES_FOLDER, str(
                image_annolist['image']['name'][0, 0][0]))

            # check if image exist
            if not os.path.isfile(image_file):
                continue
            img = Image.open(image_file)
            w, h = img.size
            img_dict = {'id': img_id, 'file_name': image_file,
                        'width': w, 'height': h}
            if mpii['img_train'][0][img_id] == 0:
                coco_train['images'].append(img_dict)
            else:
                coco_test['images'].append(img_dict)
            
            annorects = image_annolist['annorect'][0]
            # person_num
            person_num = len(annorects)

            for pid in range(person_num):
                # kps is annotated
                if check_empty(annorects[pid], 'annopoints') == False:

                    bbox = np.zeros((4))  # xmin, ymin, w, h
                    kps = np.zeros((JOINT_NUM, 3))  # xcoord, ycoord, vis

                    # kps
                    annot_joint_num = len(mpii['annolist'][0][img_id]['annorect'][0][pid]['annopoints']["point"][0][0][0])
                    for jid in range(annot_joint_num):
                        annot_jid = mpii['annolist'][0][img_id]['annorect'][0][pid]['annopoints']["point"][0][0][0][jid]['id'][0][0]
                        kps[annot_jid][0] = mpii['annolist'][0][img_id]['annorect'][0][pid]['annopoints']["point"][0][0][0][jid]['x'][0][0]
                        kps[annot_jid][1] = mpii['annolist'][0][img_id]['annorect'][0][pid]['annopoints']["point"][0][0][0][jid]['y'][0][0]
                        kps[annot_jid][2] = 1

                    # bbox extract from annotated kps
                    annot_kps = kps[kps[:, 2] == 1, :].reshape(-1, 3)
                    xmin = np.min(annot_kps[:, 0])
                    ymin = np.min(annot_kps[:, 1])
                    xmax = np.max(annot_kps[:, 0])
                    ymax = np.max(annot_kps[:, 1])
                    width = xmax - xmin - 1
                    height = ymax - ymin - 1

                    # corrupted bounding box
                    if width <= 0 or height <= 0:
                        continue
                    # 20% extend
                    else:
                        bbox[0] = (xmin + xmax)/2. - width/2*1.2
                        bbox[1] = (ymin + ymax)/2. - height/2*1.2
                        bbox[2] = width*1.2
                        bbox[3] = height*1.2

                    person_dict = {'id': aid, 'image_id': img_id, 'category_id': 1, 'area': bbox[2]*bbox[3], 'bbox': bbox.tolist(
                    ), 'iscrowd': 0, 'keypoints': kps.reshape(-1).tolist(), 'num_keypoints': int(np.sum(kps[:, 2] == 1))}
                    if mpii['img_train'][0][img_id] == 0:
                        coco_train['annotations'].append(person_dict)
                    else:
                        coco_test['annotations'].append(person_dict)
                    aid += 1

    category = {
        "supercategory": "person",
        "id": 1,  # to be same as COCO, not using 0
        "name": "person",
        "skeleton": [[0, 1],
                     [1, 2],
                     [2, 6],
                     [7, 12],
                     [12, 11],
                     [11, 10],
                     [5, 4],
                     [4, 3],
                     [3, 6],
                     [7, 13],
                     [13, 14],
                     [14, 15],
                     [6, 7],
                     [7, 8],
                     [8, 9]],
        "keypoints": ["r_ankle", "r_knee", "r_hip",
                      "l_hip", "l_knee", "l_ankle",
                      "pelvis", "throax",
                      "upper_neck", "head_top",
                      "r_wrist", "r_elbow", "r_shoulder",
                      "l_shoulder", "l_elbow", "l_wrist"]}

    coco_train['categories'] = [category]
    coco_test['categories'] = [category]

    with open(train_path, 'w') as f:
        json.dump(coco_train, f)
    
    with open(test_path, 'w') as f:
        json.dump(coco_test, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mpii-path', '-p', type=str,
                        required=True, help='path to MPII dataset root folder')
    parser.add_argument('--output-train', type=str, default='coco_train.json',
                        help='name of output file with prepared keypoints annotation')
    parser.add_argument('--output-test', type=str, default='coco_test.json',
                        help='name of output file with prepared keypoints annotation')
    args = parser.parse_args()

    convert_to_coco(args.mpii_path, args.output_train, args.output_test)
