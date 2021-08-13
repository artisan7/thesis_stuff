import pandas as pd
import cv2
import os

root_dir = 'covid-chestxray-dataset/'
image_dir = 'images/'

df = pd.read_csv('clean_data.csv')

# create directories
output_dirs = ['jonrek', 'marq', 'dwayt']
for d in output_dirs:
    try:
        os.mkdir(d)
        print('Directory', d, 'created!')
    except FileExistsError:
        print('Directory', d, 'already exists! :(')

print('Please wait while the images are being distributed...')
# distribute images evenly
current = 0
milestone = 0.1
no_of_items = df.shape[0]
for i in range(no_of_items):
    filename = df.iloc[i]['filename']
    img_loc = root_dir + image_dir + filename
    img = cv2.imread(img_loc)
    cv2.imwrite(f'{output_dirs[current % 3]}/{filename}', img)

    current = current + 1
    if current/no_of_items >= milestone:
        print(f'{round(milestone*100)}% complete...')
        milestone += .1

print('Successfully distributed images xP')