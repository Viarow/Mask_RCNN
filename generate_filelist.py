import os
from tqdm import tqdm

_UCF101_ROOT_ = '/media/Med_6T2/mmaction/data_tools/ucf101/videos/'

def average_filelist(root_path, vid_num, filelist_path):

    listfile = open(filelist_path, 'w')

    all_classes = os.listdir(root_path)
    all_classes.sort()
    for c_item in tqdm(all_classes):
        class_path = os.path.join(root_path, c_item)
        all_videos = os.listdir(class_path)

        if len(all_videos) >= vid_num:
            videos = all_videos[0: vid_num]
        else:
            print("{:d} videos in ".format(len(all_videos)) + c_item +'\n')

        for v_item in videos:
            listfile.writelines(os.path.join(c_item, v_item) + '\n')
        print("Generating video lists for " + c_item + '\n')


_Kinetics400_ROOT_ = '/media/Med_6T2/mmaction/data_tools/kinetics400/videos_val/'

def selected_filelist(selected_classes, root_path, vid_num, filelist_path):

    listfile = open(filelist_path, 'w')

    all_classes = selected_classes
    for c_item in tqdm(all_classes):
        class_path = os.path.join(root_path, c_item)
        all_videos = os.listdir(class_path)

        if len(all_videos) >= vid_num:
            videos = all_videos[0: vid_num]
        else:
            print("{:d} videos in ".format(len(all_videos)) + c_item +'\n')

        for v_item in videos:
            listfile.writelines(os.path.join(c_item, v_item) + '\n')
        print("Generating video lists for " + c_item + '\n')


def  generate_filelist():

    average_filelist(_UCF101_ROOT_,  50, './ucf101_testlist.txt')



if __name__ == '__main__':
    generate_filelist()
