from CV.AirbnbProject.DeepLearningSandbox_master.image_recognition import classify
import pandas as pd

'''
    this method is used to label the dataset. It calls classify method from DepLearningSandbox-master
    module and appends the top 3 classified objects in the dataset.
'''

def main():
    '''

    :return:
    '''
    df = pd.read_csv('E:/1RIT/CV/project/dataset/short_listings.csv')
    df = df[0:2001]
    index = 0
    for row in df.iterrows():
        url = df.iloc[index]['picture_url']
        predictions = classify.airbnbPred(url)
        for idx, item in enumerate(predictions):
            id = "id" + str(idx)
            name = "item" + str(idx)
            prob = "prob" + str(idx)
            df.at[index, id] = item[0]
            df.at[index, name] = item[1]
            df.at[index, prob] = item[2]
        print(index)
        index += 1



    df.to_csv("classifiedImage.csv")



if __name__ == '__main__':
    main()
