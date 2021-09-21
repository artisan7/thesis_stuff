import pandas as pd
import numpy as np
from skimage.io import imread
from matplotlib import pyplot as plt
from skimage import img_as_ubyte

from skimage.measure import regionprops
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy
from scipy.stats import skew

root_dir = 'preprocessed/'
img_dir = 'masked_images/'
mask_dir = 'masks/'

chunks_per_axis = 4
num_chunks = chunks_per_axis**2

features = {}
for i in range(num_chunks):
    #features[f'area{i}'] = []
    #features[f'perimeter{i}'] = []
    #features[f'eccentricity{i}'] = []
    #features[f'major axis{i}'] = []
    #features[f'minor axis{i}'] = []
    features[f'mean{i}'] = []
    features[f'variance{i}'] = []
    features[f'skewness{i}'] = []
    features[f'uniformity{i}'] = []
    features[f'snr{i}'] = []

no_of_elements = 300

for idx in range(no_of_elements):
    img_loc = f'{root_dir}{img_dir}{idx}.png'
    mask_loc = f'{root_dir}{mask_dir}{idx}.png'

    img = imread(img_loc, as_gray=True)
    #mask = imread(mask_loc, as_gray=True)

    # split into chunks
    for i in range(chunks_per_axis):
        for j in range(chunks_per_axis):
            chunk_idx = (chunks_per_axis*i) + j
            chunk_width = chunk_height = img.shape[0]//4

            img_chunk = img[ chunk_width * i:chunk_width * (i+1), chunk_height * j:chunk_height * (j+1) ]
            #mask_chunk = mask[ chunk_width * i:chunk_width * (i+1), chunk_height * j:chunk_height * (j+1) ]

            # SHAPE BASED FEATURES
            #prop = regionprops(mask)[0]

            #features[f'area{chunk_idx}'].append( prop['area'] )
            #features[f'perimeter{chunk_idx}'].append( prop['perimeter'] )
            #features[f'eccentricity{chunk_idx}'].append( prop['eccentricity'] )
            #features[f'major axis{chunk_idx}'].append( prop['major_axis_length'] )
            #features[f'minor axis{chunk_idx}'].append( prop['minor_axis_length'] )

            # UNFIROMITY, SKEWNESS, TOTAL MEAN, VARIANCE, SNR
            arr = img_chunk[img_chunk != 0] #remove zeroes (masked out pixels) from image
            
            if len(arr) > 0:
                intensity_lvls = np.unique(arr) # Get all intensity levels
                numel = intensity_lvls.size # Number of intensity levels
                features[f'uniformity{chunk_idx}'].append( numel )

                mean = np.mean(arr)
                sd = np.std(arr)
                
                features[f'mean{chunk_idx}'].append( mean )
                features[f'variance{chunk_idx}'].append( np.var(arr) )
                features[f'skewness{chunk_idx}'].append( skew(arr) )
                features[f'snr{chunk_idx}'].append( np.where(sd == 0, 0, mean/sd) )
            else:
                features[f'uniformity{chunk_idx}'].append( 0 )
                features[f'mean{chunk_idx}'].append( 0 )
                features[f'variance{chunk_idx}'].append( 0 )
                features[f'skewness{chunk_idx}'].append( 0 )
                features[f'snr{chunk_idx}'].append( 0 )

    print(f'"{idx}.png" feature extraction done')

# save as csv

df = pd.DataFrame.from_dict(features)

metadata_df = pd.read_csv('final_dataset.csv')
df['label'] = metadata_df['finding']

df.to_csv('features.csv')

print('-- FEATURE EXTRACTION FINISHED :P --')
