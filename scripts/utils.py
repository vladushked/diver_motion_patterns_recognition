import os

os.sys.path.append('../poseEstimation')
from poseEstimation.demo import infer_fast, VideoReader
from poseEstimation.modules.pose import Pose
from poseEstimation.modules.load_state import load_state
from poseEstimation.modules.keypoints import extract_keypoints, group_keypoints
from poseEstimation.models.with_mobilenet import PoseEstimationWithMobileNet

def find_dir(number, path, name):
    for dirname in os.listdir(path):
        splitted = dirname.split("-")
        if splitted[0] != name:
            continue
        if (int(splitted[1]) < number <= int(splitted[2])):
            subpath = os.path.join(path, dirname)
            for subdirname in os.listdir(subpath):
                subsplitted = subdirname.split(name)
                if subsplitted[0] != "":
                    continue
                if int(subsplitted[1]) == number:
                    dest_path = os.path.join(subpath, subdirname)
                    for dest_file in os.listdir(dest_path):
                        if dest_file.split(".")[1] == "csv":
                            yield dest_path, dest_file

def infer(net, image_provider, height_size, cpu):
    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts

    pose_sequence = []
    prediction = 0
    prediction_made = False

    for img in image_provider:
        heatmaps, pafs, scale, pad = infer_fast(
            net, img, height_size, stride, upsample_ratio, cpu)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(
                heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(
            all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (
                all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (
                all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(
                        all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(
                        all_keypoints[int(pose_entries[n][kpt_id]), 1])
                else:
                    pose_keypoints[kpt_id, 0] = 0
                    pose_keypoints[kpt_id, 1] = 0
            pose = Pose(pose_keypoints, pose_entries[n][18])
            
            current_poses.append(pose)

        if (len(current_poses) > 0):
            pose_sequence.append(current_poses[0].keypoints.reshape([36]))

    return pose_sequence