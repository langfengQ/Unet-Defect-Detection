from data_prepare import getPairs
import os
import numpy as np
import cv2

def traindata():
    data_path = os.getcwd() + os.sep + 'data' + os.sep + 'all_align_crop'
    pairs = getPairs(data_path)
    for p in pairs:
        ok = p[0]
        ng = p[1]
        msk = p[2]
        img_msk = cv2.imread(msk, cv2.IMREAD_GRAYSCALE)
        rate = np.sum(img_msk > 200)
        if rate > 30:
            img_ng = cv2.imread(ng, cv2.IMREAD_GRAYSCALE)
            cv2.imwrite((ng[:30] + '2' + ng[30:]), )


if __name__ == '__main__':


