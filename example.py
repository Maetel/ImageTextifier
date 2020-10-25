from ImageTextifier import ImageTextifier
import cv2


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


def concat_result_images(src, binarized, as_is):
    src_hi, src_wid, _ = src.shape
    bin = cv2.cvtColor(binarized, cv2.COLOR_GRAY2BGR)
    asis = cv2.cvtColor(as_is, cv2.COLOR_GRAY2BGR)
    return cv2.hconcat([src, bin, asis])

def resize(img, ratio):
    hi, wid, _ = img.shape
    return cv2.resize(img, (int(wid*ratio), int(hi*ratio)))


def main():
    # setup
    speak_process = not False
    processor = ImageTextifier()

    names = ['lena', 'itworks', 'me']
    textbox_sizes = [2, 8, 4]
    invert_options = [False, True, False]
    skip_index = []
    fill_blank_with=' ' #'.' will work too

    result_images = []

    # main loop
    for idx, name in enumerate(names):
        # load image and apply options
        if idx in skip_index:
            continue
        src = cv2.imread("img/" + name + ".jpg")
        invert_image = invert_options[idx]
        w, h = textbox_sizes[idx], textbox_sizes[idx]

        # main process
        binarized_result, binarized_text_image = processor.textify(
            src, w, h, speak_process=speak_process, invert_image=invert_image, fill_blank=fill_blank_with)
        as_is_result, as_is_text_image = processor.textify(
            src, w, h, binarize=False, speak_process=speak_process, invert_image=invert_image, fill_blank=fill_blank_with)
        
        # write result to file
        result_image = concat_result_images(
            src, binarized_text_image, as_is_text_image)
        
        result_dir = "result/"
        binarized_path, as_is_path, result_path = name + "_binarized", name + "_ as_is", name + "_result"
        write_result(binarized_result, result_dir + binarized_path + ".txt")
        write_result(as_is_result, result_dir + as_is_path + ".txt")
        cv2.imwrite(result_dir + binarized_path + ".jpg", binarized_text_image)
        cv2.imwrite(result_dir + as_is_path + ".jpg", as_is_text_image)
        cv2.imwrite(result_dir + result_path + ".jpg", result_image)

        # handle results
        print(f"Binarized result :")
        print_result(binarized_result)
        print(f"As-is result :")
        print_result(as_is_result)
        result_images.append(result_image)

    # show and wait
    for idx, result in enumerate(result_images):
        cv2.imshow('Result'+str(idx+1), result)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
