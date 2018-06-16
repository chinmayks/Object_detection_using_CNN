import cv2
import urllib.request
import numpy as np
import pandas as pd
'''
This method is used to download the dataset given by Airbnb datasets.
It reads the csv file and based on that calls the URL to fetch the image.
'''

def url_to_image(url, name, idx):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    folder = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
              6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
              11: "eleven", 12: "twelve", 13: "thirteen"}
    try:
        resp = urllib.request.urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        cv2.imwrite("C:/Users/Chinmay/PycharmProjects/cks/CV/AirbnbProject/dataset/testingdataset/"+folder[idx]+"/"+str(name)+".jpg",image)
    except Exception:
        pass


    # return the image
    #return image

df = pd.read_csv("E:/1RIT/CV/AirbnbProject/classifiedImage.csv")
df_new = pd.read_csv("C:/Users/Chinmay/PycharmProjects/cks/CV/AirbnbProject/classifiedImage.csv")
df = df[0:6149]
df.to_csv("E:/1RIT/CV/project/dataset/short_listings.csv")

for index,row in df.iterrows():
    print(index)
    url_to_image(row.loc['picture_url'], row.loc['id'], row['label'])


#url_to_image("https://a0.muscache.com/im/pictures/8ac331a9-3691-49fa-ba77-22c3cc428f1f.jpg?aki_policy=large", 18461891)