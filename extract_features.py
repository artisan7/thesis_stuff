import pandas as pd
from skimage.io import imread
from matplotlib import pyplot as plt
from skimage import img_as_ubyte

from skimage.measure import regionprops
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy

root_dir = 'preprocessed/'
img_dir = 'masked_images/'
mask_dir = 'masks/'

features = {
    'area': [],
    'perimeter': [],
    'eccentricity': [],
    'major axis': [],
    'minor axis': [],
    'contrast': [],
    'homogeneity': [],
    'energy': [],
    'correlation': [],
    'entropy': []
}

no_of_elements = 300

for idx in range(no_of_elements):
    img_loc = f'{root_dir}{img_dir}{idx}.png'
    mask_loc = f'{root_dir}{mask_dir}{idx}.png'

    img = imread(img_loc, as_gray=True)
    mask = imread(mask_loc, as_gray=True)

    # SHAPE BASED FEATURES
    prop = regionprops(mask)[0]

    features['area'].append( prop['area'] )
    features['perimeter'].append( prop['perimeter'] )
    features['eccentricity'].append( prop['eccentricity'] )
    features['major axis'].append( prop['major_axis_length'] )
    features['minor axis'].append( prop['minor_axis_length'] )

    #GLCM TEXTURE BASED FEATURES
    glcm = greycomatrix(img_as_ubyte(img), [1], [0], symmetric=True)

    features['contrast'].append( greycoprops(glcm, prop='contrast')[0][0] )
    features['homogeneity'].append( greycoprops(glcm, prop='homogeneity')[0][0] )
    features['energy'].append( greycoprops(glcm, prop='energy')[0][0] )
    features['correlation'].append( greycoprops(glcm, prop='correlation')[0][0] )

    # ENTROPY
    features['entropy'].append( shannon_entropy(img) )

    # TODO:SKEWNESS, TOTAL MEAN, VARIANCE

    print(f'"{idx}.png" feature extraction done')

# save as csv
df = pd.DataFrame.from_dict(features)
df.to_csv('features.csv')

print('-- FEATURE EXTRACTION FINISHED :P --')