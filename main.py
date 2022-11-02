# This is a sample Python script.

# Press Ctrl+F5 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import glob
import os

import numpy as np
import shutil

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press F9 to toggle the breakpoint.


def concate_name(path):
    path = path.replace('\\', '/')
    path_split = path.split("/")
    date, dir_id, file_name = path_split[-3:]
    name_cated = dir_id + "_" + date + "_" + file_name
    print(name_cated)
    # print(date, dir_id, file_name)
    # print(path_split)
    # res = os.path.basename(path)
    # print(res)
    return name_cated


def copy_file(src, dst):
    print("copy file %s --> %s" %(src, dst))
    shutil.copyfile(src, dst)

def copy_and_rename():
    root_path = "F:/fengjunxi/01_study/my_test/videos/vid-10-19-11"
    target_dir = "F:/fengjunxi/01_study/my_test/compare"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # get all vid dir
    all_paths = os.listdir(root_path)
    # print(all_paths)
    vid_res_dir = []
    for path in all_paths:
        abs_path = os.path.join(root_path, path)
        if os.path.isdir(abs_path):
            vid_res_dir.append(abs_path)

    # print(vid_res_dir)

    # get files and copy
    for vid_dir in vid_res_dir:
        # print("vid dir = ", vid_dir)
        all_files = []
        all_files.extend(glob.glob(os.path.join(vid_dir, "*.png")))
        # all_files.extend(glob.glob(os.path.join(vid_dir, "*.txt")))
        print(all_files)
        all_files.sort()
        # print(all_files)
        selected_frames = [0, 10, 30]
        # selected_frames = np.clip(selected_frames, 0, len(all_files) - 1).astype(np.int32).tolist()
        selected_frames = np.clip(selected_frames, 0, len(all_files) - 1)

        file_need_copy = []
        for index in selected_frames:
            file_need_copy.append(all_files[index])

        # file_need_copy = [all_files[0], all_files[5]]
        for src_file in file_need_copy:
            dst_file = concate_name(src_file)
            dst_file = os.path.join(target_dir, dst_file)
            copy_file(src_file, dst_file)

# IMG_EXTENSIONS = [
#     '.jpg', '.JPG','.bmp', '.BMP','.txt'
# ]
IMG_EXTENSIONS = [
    '.png','.txt'
]

VIDEO_EXTENSIONS = [
    '.mov','.MOV'
]

# Press the green button in the gutter to run the script.
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def is_video_file(filename):
    return any(filename.endswith(extension) for extension in VIDEO_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]




def get_video_path(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_video_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images



def select_frames_frome_video():
    video_dir = "./videos/"
    video_paths = get_video_path(video_dir)
    # print("vid path = ", video_paths)
    for path in video_paths:
        print("video path = ", path)
        select_frames_from_video(path)


def select_frames_from_video(video_path):
    import cv2

    cap = cv2.VideoCapture(video_path)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    # Read until video is completed
    frame_index = 0
    selected_frame_index = [3,10,30]
    selected_frames = []
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            frame_index = frame_index + 1
            if frame_index in selected_frame_index:
                selected_frames.append(frame)
                if len(selected_frames) == len(selected_frame_index):
                    break
        # Break the loop
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    for _, (index, frame) in enumerate(zip(selected_frame_index, selected_frames)):
        cv2.imwrite(video_path[:-4] + "_frame_%02d.png" % index, frame)

    print("select frames finished!")


if __name__ == '__main__':
    # select_frames_frome_video()
    copy_and_rename()
