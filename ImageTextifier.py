#from skimage.metrics import structural_similarity as ssim
import cv2 as cv
import numpy as np
from typing import List
import multiprocessing
import time
from functools import partial

ITEX_ALGO_DERIVATIVE = 0
ITEX_ALGO_BINARIZE = 1
ITEX_ALGO_AS_IS = 2

ITEX_RESOLUTION_VERY_HIGH = 250
ITEX_RESOLUTION_HIGH = 100
ITEX_RESOLUTION_MEDIUM = 60
ITEX_RESOLUTION_LOW = 30

########################################## helpers
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
# !helpers
########################################## preprocessors
def _preproc_create_texts() -> str:
    alphabets = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    nums = "0123456789"
    chars = " ,./<>?`~!@#$%^&*()-_=+[{]}\|;:\'\""
    kor = "ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㄲㄸㅃㅆㅉ"
    retval = alphabets + alphabets.lower() + nums + chars + kor
    return retval, len(retval)


def _preproc_create_text_block_image(text: str, block_wid: int = 30, block_hi: int = 30):
    if not text:
        text = ' '
    if len(text) > 1:
        text = text[0]

    # magical settings
    '''
    FONT_HERSHEY_SIMPLEX        = 0, //!< normal size sans-serif font
    FONT_HERSHEY_PLAIN          = 1, //!< small size sans-serif font
    FONT_HERSHEY_DUPLEX         = 2, //!< normal size sans-serif font (more complex than FONT_HERSHEY_SIMPLEX)
    FONT_HERSHEY_COMPLEX        = 3, //!< normal size serif font
    FONT_HERSHEY_TRIPLEX        = 4, //!< normal size serif font (more complex than FONT_HERSHEY_COMPLEX)
    FONT_HERSHEY_COMPLEX_SMALL  = 5, //!< smaller version of FONT_HERSHEY_COMPLEX
    FONT_HERSHEY_SCRIPT_SIMPLEX = 6, //!< hand-writing style font
    FONT_HERSHEY_SCRIPT_COMPLEX
    '''
    #font = cv.FONT_HERSHEY_PLAIN
    font = cv.FONT_HERSHEY_DUPLEX
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 1
    block = 30
    vertical_magic = 8
    wid, hi = block, block + vertical_magic
    img = np.zeros((hi, wid, 1), np.uint8)
    hor_offset = int(block/10)

    #x_coord = idx * block + hor_offset
    x_coord = hor_offset
    y_coord = block - hor_offset  # vertical_offset
    cv.putText(img, text,
                (x_coord, y_coord),
                font,
                fontScale,
                fontColor,
                lineType)
    img = cv.resize(img, (block_wid, block_hi))
    return img


def _preproc_create_images(block_wid: int = 30, block_hi: int = 30):
    retval = {}
    texts, textlen = _preproc_create_texts()
    for idx, text in enumerate(texts):
        img = _preproc_create_text_block_image(text, block_wid, block_hi)
        retval[texts[idx]] = img
    return retval

def _preproc_binarize(src):
    img = src[:]
    thres, otsu = cv.threshold(
            img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        # showimg(otsu)
    for row, row_data in enumerate(otsu):
        for col, pixel_value in enumerate(row_data):
            if not pixel_value:
                img[row][col] = 0
    return img

def _preproc_derivative(src):
    hi, wid = src.shape
    img = np.zeros((hi, wid, 1), np.uint8)
    #der = cv.Scharr(src, cv.CV_64F, 1, 0)
    #der = cv.Laplacian(src, cv.CV_64F)
    der = cv.Canny(src, 100, 100)
    vf = np.vectorize(lambda x : abs(x))
    vf(der)
    
    min, max = np.amin(der), np.amax(der)
    thres = 0#max*112/113
    vf = np.vectorize(lambda x: 0 if x < thres else x)
    vf(der)
    #dif = max - min
    img = (((der-thres)/(max-thres)) * 255).astype(np.uint8)

    bin = _preproc_binarize(img)

    #showimg(der)
    #showimg(bin)

    #for row, row_data in enumerate(scharr):
    #    for col, pixel_value in enumerate(row_data):
    return bin

def preprocess_source(src, algorithm, invert=True):
    retval = None
    img = src[:]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if invert:
        gray = (255-gray)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    if algorithm == ITEX_ALGO_BINARIZE:
        retval = _preproc_binarize(blurred)
    elif algorithm == ITEX_ALGO_AS_IS:
        retval = blurred
    else: #default
        retval = _preproc_derivative(blurred)
        
    return retval
#!preprocessors
########################################## Pure functions

def block_idx_to_img(img, block_hor, block_wid, block_hi, comparer, fill_blank, block_idx):
    col = block_idx % block_hor
    row = block_idx // block_hor

    block = img[row*block_hi: (row+1)*block_hi,
                col*block_wid: (col+1)*block_wid]
    text, score = comparer(block, fill_blank)
    return text

# !Pure functions
########################################## Main class
class ImageTextifier:
    def __init__(self, block_wid: int = 20, block_hi: int = 20):
        self.dataset = None
        self.create_dataset(block_wid, block_hi)

    def create_dataset(self, block_wid: int = 20, block_hi: int = 20):
        self.dataset = _preproc_create_images(block_wid, block_hi)
        return self.dataset

    # match dataset size to source block size
    def update_dataset_size(self, src_block):
        block_hi, block_wid = src_block.shape
        self.update_dataset_size(block_hi, block_wid)

    def update_dataset_size(self, block_hi, block_wid):
        if not self.dataset:
            return
        datum_hi, datum_wid = self.dataset['A'].shape
        if (block_wid, block_hi) != (datum_wid, datum_hi):
            for k, v in self.dataset.items():
                self.dataset[k] = cv.resize(v, (block_wid, block_hi))

    # compare a small image to a set of text images, and returns a character that matches
    def compare_block(self, src_block, fill_blank = ' '):
        if not self.dataset:
            return
        #if empty block, return fill_blank with the highest score
        if not np.sum(src_block):
            return fill_blank, 1

        # assume dataset size is already updated
        #self.update_dataset_size(src_block)
        highest_score = 0
        highest_text = fill_blank
        for k, v in self.dataset.items():
            # similarity = ssim(v, src_block) #skimage
            '''
            matchTemplate methods :
            cv.TM_CCOEFF
            cv.TM_CCOEFF_NORMED
            cv.TM_CCORR
            cv.TM_CCORR_NORMED
            cv.TM_SQDIFF
            cv.TM_SQDIFF_NORMED
            '''
            similarity = cv.matchTemplate(v, src_block, cv.TM_CCOEFF_NORMED)
            if highest_score < similarity:
                highest_score = similarity
                highest_text = k
                #print(f"Highest score/text : {highest_score}/{highest_text}")
        return highest_text, highest_score



    # main method
    def textify(self, src, grid_size=ITEX_RESOLUTION_MEDIUM, algorithm = ITEX_ALGO_DERIVATIVE, speak_process=True, speak_result_as_text=True, return_text_image=True, invert_image=False, fill_blank = ' '):
        if speak_process:
            print("Warming up...")
        #setup basic variables
        hi_src, wid_src, _ = src.shape
        wid_dst, hi_dst = wid_src - wid_src % grid_size, hi_src - hi_src % grid_size
        smaller_block_size = min(wid_dst // grid_size, hi_dst // grid_size)
        block_wid, block_hi = smaller_block_size, smaller_block_size
        block_hor, block_ver = wid_dst // block_wid, hi_dst // block_hi
        block_count = block_hor * block_ver
        speak_process_every_nth_block = 1 if block_count <= 13 else int(block_count // 13)
        self.update_dataset_size(block_hi, block_wid)

        img = cv.resize(src[:], (wid_dst, hi_dst))
        img = preprocess_source(img, algorithm, invert_image)
        
        if not fill_blank:
            fill_blank = ' '
        elif len(fill_blank) > 1:
            fill_blank = fill_blank[0]
        
        # begin main processing
        if speak_process:
            print("Processing...")
        _begin = time.perf_counter()
        manager = multiprocessing.Manager()
        func = partial(block_idx_to_img, img, block_hor, block_wid, block_hi, self.compare_block, fill_blank)
        pool = multiprocessing.Pool()
        retval = pool.map(func, range(block_count))
        pool.close()
        pool.join()
        _end = time.perf_counter()
        if speak_process:
            print(f"Finished ({_end - _begin:0.4f})s")

        # handle results
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

# !Main class
