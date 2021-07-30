import os, shutil

import cv2

root = '/home/wan/datasets/CAMPUS'
scenarios = ['Auditorium', 'Garden1', 'Garden2', 'Parkinglot']

def get_trackes(scenario, video_name):
    frame_track = {}

    with open(os.path.join('Campus', scenario, video_name+'.txt')) as f:
        lines = f.readlines()

        for line in lines:
            line = line.strip()

            track_id, x_min, y_min, x_max, y_max, frame_id, lost, occluded, generated = map(int, line.split(' ')[:-1])

            # if lost == 1 or occluded == 1:
            if lost == 1:
                continue

            if frame_id not in frame_track.keys():
                frame_track[frame_id] = []

            frame_track[frame_id].append({
                'track_id': track_id,
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max,
                'generated': generated
            })
    
    return frame_track

def extract_video(scenario, video_name):
    print(f'Extracting scenario {scenario}, {video_name}')

    os.makedirs(os.path.join('Campus', scenario, video_name), exist_ok=True)

    cap = cv2.VideoCapture(os.path.join(root, scenario, video_name+'.mp4'))

    frame_track = get_trackes(scenario, video_name)

    cur_frame_id = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if cur_frame_id in frame_track.keys():

            for bbox in frame_track[cur_frame_id]:
                track_id = bbox['track_id']
                x_min = bbox['x_min']
                y_min = bbox['y_min']
                x_max = bbox['x_max']
                y_max = bbox['y_max']
                generated = bbox['generated']

                image = frame[y_min:y_max+1,x_min:x_max+1]

                image_id = '{:05d}_{:05d}'.format(track_id, cur_frame_id)

                cv2.imwrite(os.path.join('Campus', scenario, video_name, image_id+'.jpg'), image)

        cur_frame_id += 1


def main():

    os.makedirs('Campus', exist_ok=True)

    for scenario in scenarios:

        os.makedirs(os.path.join('Campus', scenario), exist_ok=True)

        files = os.listdir(os.path.join(root, scenario))

        video_files = []

        for filename in files:
            if filename.split('.')[-1] == 'mp4':
                video_files.append(filename)
            if filename.split('.')[-1] == 'txt':
                shutil.copy2(os.path.join(root, scenario, filename), os.path.join('Campus', scenario, filename))
        
        for video_file in video_files:
            extract_video(scenario, video_file.split('.')[0])

if __name__=='__main__':
    main()