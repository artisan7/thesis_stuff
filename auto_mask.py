import pandas as pd
from skimage import io
from matplotlib import pyplot as plt
from skimage.transform import resize
from skimage import img_as_ubyte
import skimage.filters as flt
import skimage.exposure as expo
from skimage.morphology import square, dilation, erosion, opening, closing, remove_small_objects, remove_small_holes

root_dir = ''
image_dir = 'final_cxr_dataset/'

df = pd.read_csv('final_dataset.csv')

for i in range(300):
    length = 200

    current = df.iloc[i]
    img_loc = root_dir + image_dir + current['filename']

    coarse_mask = img_as_ubyte(io.imread('coarse_mask.png', as_gray=True))

    try:
        img = img_as_ubyte(io.imread(img_loc, as_gray=True))
        y_len, x_len = img.shape

        if y_len > x_len:    # height > width
            img = resize(img, [ round(length*y_len/x_len), length ])
            y_len = img.shape[0]
            crop_length = y_len - length
            img = img[ :y_len-crop_length , : ]    # crop from bottom

        elif y_len < x_len:    # height < width

            img = resize(img, [ length, round(length*x_len/y_len) ])
            x_len = img.shape[1]
            crop_length = x_len - length
            half_crop = crop_length//2
            img = img[ : , half_crop:x_len-half_crop ]    # crop on left and right

            if img.shape[0] != img.shape[1]:
                img = img[ : , :length ]    # make sure image is square

        else:    # height == width
            img = resize(img, [ length, length ])

        img = flt.gaussian(img)
    except FileNotFoundError:
        print(f'{img_loc} not found.')
        
    
    #coarse mask -> thresholding
    masked = ~(img_as_ubyte(img) * coarse_mask)
    thresh = flt.threshold_yen(masked)
    binary1 = masked <= thresh

    # thresholding -> coarse mask
    #thresh = flt.threshold_yen(img_as_ubyte(img))
    #masked = ~(img_as_ubyte(img) * coarse_mask)
    #binary2 =  masked <= thresh

    kernel = square(7)
    op = opening(binary1, kernel)#select binary here
    #op = dilation(remove_small_objects(erosion(op, kernel), 800), kernel)
    mask = closing(op, kernel)
    #mask = remove_small_objects(closing(op, kernel), 1000)
    mask = remove_small_holes(closing(op, kernel), 1000)
    final1 = img * mask

    plt.imshow(final1, cmap=plt.cm.gray)



    output_dir = 'preprocessed/'
    img_dir = 'masked_images/'
    mask_dir = 'masks/'
    filename = f'{i}.png'

    io.imsave(f'{output_dir}{img_dir}{filename}', img_as_ubyte(final1))
    io.imsave(f'{output_dir}{mask_dir}{filename}', img_as_ubyte(mask))
    print(f'"{filename}" saved')