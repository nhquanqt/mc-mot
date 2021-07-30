import os

import torch
from torch.utils.data import Dataset

import cv2

def get_trackes(path):
    frame_track = {}

    images = os.listdir(path)
    images = sorted(images)

    for filename in images:
        track_id = int(filename.split('.')[0].split('_')[0])
        frame_id = int(filename.split('.')[0].split('_')[1])

        if frame_id not in frame_track.keys():
            frame_track[frame_id] = []

        frame_track[frame_id].append(track_id)

    return frame_track

class Campus(Dataset):
    def __init__(self, root):
        super(Campus, self)

        self.scenarios = ['Auditorium', 'Garden1', 'Garden2', 'Parkinglot']

        self.scenario_to_id = {self.scenarios[i]:i for i in range(4)}

        self.tracks = {}

        for scenario in self.scenarios:
            files = os.listdir(os.path.join(root, scenario))

            scenario_id = self.scenario_to_id[scenario]

            self.tracks[scenario_id] = {}

            video_names = []

            for filename in files:
                if filename.split('.')[-1] == 'txt':
                    video_names.append(filename.split('.')[0])

            video_names = sorted(video_names)

            for i, video_name in enumerate(video_names):

                self.tracks[scenario_id][i] = get_trackes(os.path.join(root, scenario, video_name))

    def get_tracks(self, frame_id, scenario=None, scenario_id=None):
        assert scenario is not None or scenario_id is not None

        if scenario_id is None:
            scenario_id = self.scenario_to_id[scenario]

        ret = []

        for i in range(4):
            if frame_id in self.tracks[scenario_id][i].keys():
                ret.append(self.tracks[scenario_id][i][frame_id])
            else:
                ret.append([])

        return ret


if __name__ == '__main__':
    dataset = Campus('/home/wan/Thesis/project/mc-mot-baseline/data/Campus')

    ret = dataset.get_tracks(412, scenario_id=2)

    print(ret)
