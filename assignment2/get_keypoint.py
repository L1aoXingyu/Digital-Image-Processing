import os
import sys

import dlib

import cv2
from utils import draw_point

img_name = sys.argv[1]
img_root = os.path.join(os.getcwd(), img_name)
img = cv2.imread(img_root)


def get_dot(event, x, y, flags, param):
    global img_name
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 10, (255, 0, 0), -1)
        with open('./{}.txt'.format(img_name.split('.')[0]), 'a') as f:
            f.write('{}, {}\n'.format(x, y))


def main():
    global img
    global img_name
    # 建立dlib中的面部特征监测器，并读入预训练的参数
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

    dets = detector(img, 1)
    key_point = []
    for i, d in enumerate(dets):
        print('Detection {}, left {}, top {} right {} bottom {}'.format(
            i, d.left(), d.top(), d.right(), d.bottom()))
        shape = predictor(img, d)
        for p in shape.parts():
            key_point.append((p.x, p.y))
            with open('./{}.txt'.format(img_name.split('.')[0]), 'a') as f:
                f.write('{}, {}\n'.format(p.x, p.y))

    for p in key_point:
        draw_point(img, p, (255, 0, 0))

    cv2.namedWindow(img_name, 0)
    cv2.setMouseCallback(img_name, get_dot)
    while (1):
        cv2.resizeWindow(img_name, 800, 1000)
        cv2.imshow(img_name, img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    cv2.imwrite('new_{}.jpg'.format(img_name.split('.')[0]), img)


if __name__ == '__main__':
    main()
