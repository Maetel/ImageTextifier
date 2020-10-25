import numpy as np
import cv2
from typing import List


def createTexts() -> str:
    alphabets = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    nums = "0123456789"
    chars = " ,./<>?`~!@#$%^&*()-_=+[{]}\|;:\'\""
    retval = alphabets + alphabets.lower() + nums + chars
    return retval, len(retval)


def create_text_block_image(text: str, block_wid: int = 30, block_hi: int = 30):
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
    #font = cv2.FONT_HERSHEY_PLAIN
    font = cv2.FONT_HERSHEY_DUPLEX
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
    cv2.putText(img, text,
                (x_coord, y_coord),
                font,
                fontScale,
                fontColor,
                lineType)
    img = cv2.resize(img, (block_wid, block_hi))
    return img


def createImages(block_wid: int = 30, block_hi: int = 30):
    retval = {}
    texts, textlen = createTexts()
    for idx, text in enumerate(texts):
        img = create_text_block_image(text, block_wid, block_hi)
        retval[texts[idx]] = img
    return retval


def main():
    images = createImages(30, 30)
    show_until = 2
    for idx, key in enumerate(images):
        cv2.imshow("img", images[key])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if idx >= show_until:
            break


if __name__ == "__main__":
    main()
