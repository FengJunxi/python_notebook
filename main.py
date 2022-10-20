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
    print("copy file %s --> %s", src, dst)
    shutil.copyfile(src, dst)

def copy_and_rename():
    root_path = "F:/fengjunxi/01_study/my_test/images/vid-10-20-20"
    target_dir = "F:/fengjunxi/01_study/my_test/images/compare"
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
        all_files = glob.glob(os.path.join(vid_dir, "*.txt"))
        all_files.sort()
        # print(all_files)
        file_need_copy = [all_files[0], all_files[5]]
        for src_file in file_need_copy:
            dst_file = concate_name(src_file)
            dst_file = os.path.join(target_dir, dst_file)
            copy_file(src_file, dst_file)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')
    copy_and_rename()

