import json
import skimage
from skimage import io
from skimage.transform import resize
from matplotlib import pyplot as plt
from skimage.draw import polygon2mask
import numpy as np
from functools import reduce
from skimage.filters import gaussian

# load marks
f = open('manual_marks.json', 'r')
data = json.load(f)

metadata = data['_via_img_metadata']
marked = []
for key, value in metadata.items():
    if len(metadata[key]) > 0:
        marked.append(metadata[key])

# restructure data for easier access
def rearrange_data(img_data):
    polygons = []
    
    for r in img_data['regions']:
        points = []
        
        attribs = r['shape_attributes']
        for i in range(len(attribs['all_points_x'])):
            points.append([ attribs['all_points_y'][i], attribs['all_points_x'][i] ])
        
        polygons.append(points)

    return {'filename': img_data['filename'], 'polygons': polygons}

d = list(map(rearrange_data, marked))   # new data structure
features = {}
for x in d:
    # load image
    polygon_data = x
    crop_w = 250

    img_loc = f'covid-chestxray-dataset/images/{polygon_data["filename"]}'
    img = io.imread(img_loc, as_gray=True)
    orig_shape = img.shape
    img = resize(img, [ round(crop_w*img.shape[0]/img.shape[1]), crop_w ])
    plt.imshow(img, cmap=plt.cm.gray)

    # load mask
    mask = []
    for p in polygon_data['polygons']:
        mask.append(polygon2mask(orig_shape, p))
    if len(mask) < 2:
        break;
    mask = reduce(np.bitwise_or, mask)
    mask = resize(mask, img.shape )

    plt.imshow(mask, cmap=plt.cm.gray)

    # gaussian blur
    blurred = gaussian(img, sigma=3)
    masked_img = skimage.img_as_int(blurred*mask)
    plt.imshow(masked_img, cmap=plt.cm.gray)
    masked_img.max()

    # get shape based features through regionprops
    from skimage.measure import regionprops
    #from skimage.feature import greycomatrix, greycoprops

    props = regionprops(masked_img)
    #glcm = greycomatrix(masked_img, [1], [0, np.pi/2], levels=21288)
    #gray_props = greycoprops(glcm)

    # extract features
    f = {
        'area': 0,
        'perimeter': 0,
        'eccentricity': 0,
    }
    for p in props:
        f['area'] += p.area
        f['perimeter'] += p.perimeter
        f['eccentricity'] += p.eccentricity

    f['eccentricity'] = f['eccentricity']/len(props)
    features[polygon_data["filename"]] = f
    print(f'"{polygon_data["filename"]}" feature extration success')


#convert and save to json
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NpEncoder, self)

with open('features.json', 'w') as outfile:
    json.dump(features, outfile, cls=NpEncoder)