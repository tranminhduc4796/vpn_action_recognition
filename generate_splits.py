import argparse
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split


CLASSES_ID = {'008', '009', '027', '058', '059', '060'}


def parse_args():
    parser = argparse.ArgumentParser(description='Generate split txt for train & test')

    parser.add_argument('--video_path', type=str,   help='directory stores videos')
    parser.add_argument('--skeleton_npy_path', default=250, type=int, help='directory stores skeleton npy')
    parser.add_argument('--out', type=str, default='splits', help='directory stores 2 train.txt & valid.txt')
    return parser.parse_args()


def create_dataframe(skeleton_dir, video_dir):
    df = pd.DataFrame(columns=['filepath'])

    for file_path in tqdm(os.listdir(skeleton_dir)):
        video_path = os.path.join(video_dir, file_path[:-4])
        has_action = file_path[file_path.find('A') + 1:file_path.find('A') + 4] in CLASSES_ID
        if has_action:
            if os.path.exists(video_path):
                npy_data = np.load(os.path.join(skeleton_dir, file_path), allow_pickle=True)
                equal_length = len(npy_data) == len(os.listdir(video_path))
                if not equal_length:
                    print('Number of data is different number of frames', file_path)
                else:
                    df = df.append({'filepath': file_path[:-4]}, ignore_index=True)
            else:
                print('No video for', file_path)
    df['action'] = df['filepath'].str[-3:].apply(int)
    return df


def generate_data(df, out_path):
    x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, 0], df.iloc[:, 1], test_size=0.15,
                                                        random_state=1,
                                                        stratify=df.iloc[:, 1])
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2,
                                                      random_state=1, stratify=y_train)

    np.savetxt(os.path.join(out_path, 'train.txt'), x_train.to_numpy(), fmt='%s')
    np.savetxt(os.path.join(out_path, 'test.txt'), x_test.to_numpy(), fmt='%s')
    np.savetxt(os.path.join(out_path, 'valid.txt'), x_val.to_numpy(), fmt='%s')


if __name__ == '__main__':
    args = parse_args()
    skeleton_df = create_dataframe(args.skeleton_npy_path, args.video_path)
    generate_data(skeleton_df, args.out)
