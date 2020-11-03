import ImageTextifier as ITEX
import cv2
import sys

########################################## helpers
def is_gui():
    return not sys.stdin.isatty()


def in_between(src:str, chars:str = "[]"):
    if len(chars) == 1:
        return src[:src.rfind(chars)]
    elif len(chars) > 2:
        chars = chars[:2]
    end = src.rfind(chars[1])
    start = src[:end].rfind(chars[0])+1
    return src[start:end]
    
def print_result(list):
    for row in list:
        print("".join(row))


def write_result(list, path, carrige_return=True):
    file = open(path, "w")
    for row in list:
        file.write("".join(row))
        if carrige_return:
            file.write('\n')
    file.close()

def concat_result_images(src, der_img, bin_img):
    return cv2.hconcat([src, cv2.cvtColor(der_img, cv2.COLOR_GRAY2BGR), cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)])

def resize(img, ratio):
    hi, wid, _ = img.shape
    return cv2.resize(img, (int(wid*ratio), int(hi*ratio)))


########################################## Examples begin here

def simple_example():
    print("simple_example() begins...")

    image_path = "img/lena.jpg"
    textifier = ITEX.ImageTextifier()
    result_text, result_image = textifier.textify(cv2.imread(image_path))

    cv2.imshow("Simple example result", result_image)
    cv2.waitKey(0)
    print("simple_example() finished")

def full_function_example():
    print("full_function_example() begins...")

    image_path = "img/lena.jpg"
    src_image = cv2.imread(image_path)
    image_name = in_between(image_path, '/.')
    textifier = ITEX.ImageTextifier()

    # setup
    algorithm1 = ITEX.ITEX_ALGO_DERIVATIVE
    algorithm2 = ITEX.ITEX_ALGO_BINARIZE
    grid_size = ITEX.ITEX_RESOLUTION_LOW #equals to int(30)
    invert_image = False
    speak_process = True
    speak_result_as_text = False
    return_text_image = True
    fill_blank_with=' ' #' ' will work too

    derivative_result, derivative_text_image = textifier.textify(src_image, grid_size=grid_size, speak_process=speak_process, algorithm=algorithm1, invert_image=invert_image, speak_result_as_text=speak_result_as_text, return_text_image=return_text_image, fill_blank=fill_blank_with)

    binarized_result, binarized_text_image = textifier.textify(src_image, grid_size=grid_size, speak_process=speak_process, algorithm=algorithm2, invert_image=invert_image, speak_result_as_text=speak_result_as_text, return_text_image=return_text_image, fill_blank=fill_blank_with)

    #show results
    print(f"Algorithm - Derivative :")
    print_result(derivative_result)
    print(f"Algorithm - Binarized :")
    print_result(binarized_result)
    
    concatted = concat_result_images(src_image, derivative_text_image, binarized_text_image)
    if is_gui():
        cv2.imshow("Full function result", concatted)
    cv2.waitKey(0)

    print("full_function_example() finished")


if __name__ == "__main__":
    #simple_example()
    full_function_example()
