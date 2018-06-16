import pandas as pd
import numpy as np
import operator
from CV.AirbnbProject import imageDownload
import urllib.request

'''
    This method downloads all the images based on wnnid from ImageNet and stores it 
    in different subfolders.

'''
def findTopLabels(df):
    '''
    find top labels in the dataset
    :param df:  dataframe
    :return: dict of labels
    '''

    labels = {}
    for idx, row in df.iterrows():
        if row['item0'] in labels:
            labels[row['item0']] += 1
        else:
            labels[row['item0']] = 1

        # if row['item1'] in labels:
        #     labels[row['item1']] += 1
        # else:
        #     labels[row['item1']] = 1
        # if row['item2'] in labels:
        #     labels[row['item2']] += 1
        # else:
        #     labels[row['item2']] = 1
    labels = sorted(labels.items(), key = operator.itemgetter(1), reverse= True)
    labels = labels[0:13]
    labelID = {}
    for idx, items in enumerate(labels):
        labelID[items[0]] = idx +1
    print(labelID)
    for idx, row in df.iterrows():
        if row['item0'] in labelID:
            df.at[idx, 'label'] = int(labelID[row['item0']])

        elif row['item1'] in labelID:
            df.at[idx, 'label'] = int(labelID[row['item1']])

        elif row['item2'] in labelID:
            df.at[idx, 'label'] = int(labelID[row['item2']])

        else:
            df.at[idx, 'label'] = np.NAN
    df = df[np.isfinite(df['label'])]
    return labels, df

def getImageURLsOfWnid(wnid):
    url = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=' + str(wnid)
    f = urllib.request.urlopen(url)
    contents = str(f.read()).split('\\r\\n')
    imageUrls = []

    for each_line in contents:
        # Remove unnecessary char
        each_line = each_line.replace('\r', '').strip()
        if each_line:
            imageUrls.append(each_line)

    return imageUrls

def main():
    ''''
        Download image from ImageNet
    '''
    df = pd.read_csv('C:/Users/Chinmay/PycharmProjects/cks/CV/AirbnbProject/classifiedImage.csv')
    labels, df = findTopLabels(df)
    df.to_csv("classifiedImage.csv", index= None)
    wnetID = {'studio_couch':'n04344873', 'bookcase':'n02870880', 'four-poster':'n03388549','quilt':'n04033995',
              'patio':'n03899768', 'home_theater':'n03529860', 'dining_table':'n03201208',
              'window_shade':'n04590129','wardrobe':'n04550184','sliding_door':'n04239074','desk':'n03179701',
              'restaurant':'n04081281','microwave':'n03761084'}
    for k,v in wnetID.items():
        urls = getImageURLsOfWnid(v)
        for idx, url in enumerate(urls):
            imageDownload.url_to_image(url, k,idx)
            if idx == 600:
                break



if __name__ == '__main__':
    main()