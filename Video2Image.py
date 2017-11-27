import os
import cv2

TRAIN_ROOT = '/home/lenovo/yql/pig_data/train_folder'
VALIDATION_ROOT = '/home/lenovo/yql/pig_data/validation_folder/'
videos_folder = '/home/lenovo/yql/pig_data/videos_folder/'


def video2image(video):
    vc = cv2.VideoCapture(video)
    c = 1
    train_count = 1
    valid_count = 1
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False
        print('The viode is not found')
    video_path, _ = os.path.splitext(video)
    video_name = video_path.split('/')[-1]
    timeF = 10
    while rval:
        rval, frame = vc.read()
        if c % 100 == 0: #save validation data
            print(os.path.join(VALIDATION_ROOT, 'image%02d-%04d.jpg' % (int(video_name), valid_count)))
            cv2.imwrite(os.path.join(VALIDATION_ROOT, 'image%02d-%08d.jpg' % (int(video_name), valid_count)), frame)
            valid_count += 1
        elif c % 5 == 0: # save train data
            print(os.path.join(TRAIN_ROOT, 'image%02d-%04d.jpg' % (int(video_name), train_count)))
            cv2.imwrite(os.path.join(TRAIN_ROOT, 'image%02d-%08d.jpg' % (int(video_name), train_count)), frame)
            train_count += 1
        c = c + 1
        cv2.waitKey(1)
    vc.release()

if __name__ == '__main__':

    videos = os.listdir('/home/lenovo/yql/pig_data/videos_folder')
    for file_name in videos:
        video_path = os.path.join('/home/lenovo/yql/pig_data/videos_folder', file_name)
        print(video_path)
        video2image(video_path)
        print('video %s is capture ok' % (file_name))