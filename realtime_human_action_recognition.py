
import numpy as np
import cv2
import os
import torch
import torch.nn as nn

os.sys.path.append('poseEstimation')
from poseEstimation.models.with_mobilenet import PoseEstimationWithMobileNet
from poseEstimation.modules.keypoints import extract_keypoints, group_keypoints
from poseEstimation.modules.load_state import load_state
from poseEstimation.modules.pose import Pose
from poseEstimation.demo import infer_fast, VideoReader


LABELS = ["Around",
          "ComeHere",
          "Danger You",
          "DontKnow",
          "OKsurface",
          "Over Under",
          "Think PressureBalancePb ReserveOpened",
          "Watch",
          "CannotOpenReserve",
          "Cold",
          "Help",
          "Me",
          "Meet",
          "OutOfAir",
          "Stop"]

N_CLASSES = 15
INPUT_DIM = 36
SEQUENCE_LENGTH = 16


class LstmClassifier(nn.Module):
    def __init__(self, input_dim, n_classes, lstm_hidden_dim=256, fc_hidden_dim=256, n_lstm_layers=2):
        super(LstmClassifier, self).__init__()

        self._lstm = nn.LSTM(input_size=input_dim,
                             hidden_size=lstm_hidden_dim,
                             num_layers=n_lstm_layers,
                             batch_first=True)

        self._fc = nn.Sequential(nn.Linear(lstm_hidden_dim, fc_hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(fc_hidden_dim, n_classes))

    def forward(self, x):
        lstm_output, _ = self._lstm.forward(x)
        lstm_output = lstm_output[:, -1, :]
        fc_output = self._fc.forward(lstm_output)
        return fc_output


def classificateAction(model, sequence, device):
    model.eval()
    sequence = sequence.to(device)

    with torch.no_grad():
        y_pred = model.forward(sequence)
    return torch.argmax(y_pred, dim=1), y_pred


def infer(net, image_provider, height_size, cpu, device, model):
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
            # print(current_poses[0].keypoints.reshape([1,36]))
            pose_sequence.append(current_poses[0].keypoints.reshape([36]))
            # print(len(pose_sequence))
            # print(type(current_poses[0].keypoints))

        if (len(pose_sequence) == 32):
            # print(pose_sequence)
            sequence = torch.FloatTensor(pose_sequence).unsqueeze(0)
            # print(sequence.shape)
            prediction, values = classificateAction(model, sequence, device)
            print(values)
            if (values[0][prediction].item() > 1.0):
                prediction_made = True
            else:
                prediction_made = False
            pose_sequence = pose_sequence[-4:]

        if (prediction_made):
            cv2.putText(img, LABELS[int(prediction)], (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        for pose in current_poses:
            pose.draw(img)
        # img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)

        cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
        key = cv2.waitKey(1)
        if key == 27:  # esc
            return


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # DEVICE = "cpu"
    print("Using device: " + DEVICE)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.enabled = True

    SEED = 42

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(
        "weights/checkpoint_iter_370000.pth", map_location='cpu')
    load_state(net, checkpoint)

    model = LstmClassifier(INPUT_DIM, N_CLASSES)
    model = model.to(DEVICE)
    model.load_state_dict(torch.load("lstm_action_classifier.pth.tar"))

    frame_provider = VideoReader("0")
    infer(net, frame_provider, 256, False, DEVICE, model)
