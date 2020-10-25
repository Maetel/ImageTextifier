#from skimage.metrics import structural_similarity as ssim
import ImageTextifierPreproc as preprocess
import cv2 as cv
import numpy as np
from typing import List

# helpers


def showimg(img):
    cv.imshow("temp", img)
    cv.waitKey(0)


def array_to_2D(list: List, stride: int):
    if (len(list) % stride) != 0:
        raise Exception(
            f"list(length:{len(list)}) not divisible by stride({stride})")
        return None
    retval = []
    rows = len(list) // stride
    for row in range(rows):
        start, end = row * stride, (row+1) * stride
        retval.append(list[start:end])
    return retval


def preprocess_source(src, binarize=True, invert=True):
    retval = None
    img = src[:]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if invert:
        gray = (255-gray)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    if binarize:
        thres, otsu = cv.threshold(
            blurred, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        # showimg(otsu)
        for row, row_data in enumerate(otsu):
            for col, pixel_value in enumerate(row_data):
                if not pixel_value:
                    blurred[row][col] = 0
        retval = blurred
    else:
        retval = blurred
    return retval

# Main class


class ImageTextifier:
    def __init__(self, block_wid: int = 20, block_hi: int = 20):
        self.dataset = None
        self.create_dataset(block_wid, block_hi)

    def create_dataset(self, block_wid: int = 20, block_hi: int = 20):
        self.dataset = preprocess.createImages(block_wid, block_hi)
        return self.dataset

    # match dataset size to source block size
    def update_dataset_size(self, src_block):
        if not self.dataset:
            return
        block_hi, block_wid = src_block.shape
        datum_hi, datum_wid = self.dataset['A'].shape
        if (block_wid, block_hi) != (datum_wid, datum_hi):
            for k, v in self.dataset.items():
                self.dataset[k] = cv.resize(v, (block_wid, block_hi))

    # compare a small image to a set of text images, and returns a character that matches
    def compare_block(self, src_block, fill_blank = ' '):
        if not self.dataset:
            return

        self.update_dataset_size(src_block)
        highest_score = 0
        highest_text = fill_blank
        for k, v in self.dataset.items():
            # similarity = ssim(v, src_block) #skimage
            similarity = cv.matchTemplate(v, src_block, cv.TM_CCOEFF)
            if highest_score < similarity:
                highest_score = similarity
                highest_text = k
                #print(f"Highest score/text : {highest_score}/{highest_text}")
        return highest_text, highest_score

    # main method
    def textify(self, src, block_wid: int = 20, block_hi: int = 20, speak_process=True, binarize=True, speak_result_as_text=True, return_text_image=True, invert_image=False, fill_blank = ' '):
        hi_src, wid_src, _ = src.shape
        wid_dst, hi_dst = wid_src - wid_src % block_wid, hi_src - hi_src % block_hi

        img = cv.resize(src[:], (wid_dst, hi_dst))
        img = preprocess_source(img, binarize, invert_image)
        block_hor, block_ver = wid_dst // block_wid, hi_dst // block_hi
        block_count = block_hor * block_ver
        speak_process_every_nth_block = 1 if block_count <= 17 else int(
            block_count // 17)

        if not fill_blank or len(fill_blank) > 1:
            fill_blank = ' '
        retval = [fill_blank for cnt in range(block_count)]

        if speak_process:
            print("Processing...")

        for block_idx in range(block_count):
            if speak_process:
                if (block_idx % speak_process_every_nth_block) == 0:
                    print(f"  {int((block_idx/block_count)*100)}%")
            col = block_idx % block_hor
            row = block_idx // block_hor

            block = img[row*block_hi: (row+1)*block_hi,
                        col*block_wid: (col+1)*block_wid]
            text, score = self.compare_block(block, fill_blank)
            retval[block_idx] = text

        if speak_process:
            print("Finished")

        if speak_result_as_text:
            for row in range(block_ver):
                start = block_ver * row
                print("".join(retval[start: start + block_hor]))

        retval_2D = array_to_2D(retval, block_hor)

        if return_text_image:
            text_image = np.zeros_like(img)
            for block_idx, char in enumerate(retval):
                col = block_idx % block_hor
                row = block_idx // block_hor
                text_image[row*block_hi: (row+1)*block_hi,
                           col*block_wid: (col+1)*block_wid] = self.dataset[char]
            text_image = cv.resize(text_image, (wid_src, hi_src))
            return retval_2D, text_image

        return retval_2D
