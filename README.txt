README: 
1. Preprocessing Files:
	editCSV.py
	featureExtraction.py
	imageDownload.py
	imageDownloadfromImageNet.py

2.	Model:
	Model_creation.py


Setup
	Download and install Tensorflow, keras, Pillow as PIL, openCV, numpy, pandas, glob

	No need to run all the pre processing files:
	Download dataset from thin link
	link to the dataset 'https://drive.google.com/open?id=1pJl0_06-1Hci7khqScrRy7cg-5UUalMK'

	Run only model_creation.py
	Before running this file it needs sligh editing for setting the file path:

	Set these 2 path names and then run the program
		pathofTrainingDataset = 'C:/Users/Chinmay/PycharmProjects/cks/CV/AirbnbProject/'
		pathofTestingDataset =  'C:/Users/Chinmay/PycharmProjects/cks/CV/AirbnbProject/'

References for this project:


1. https://deeplearningsandbox.com/how-to-build-an-image-recognition-system-using-keras-and-tensorflow-for-a-1000-everyday-object-559856e04699 (Used to impute label)
2  https://machinelearningmastery.com/object-recognition-convolutional-neural-networks-keras-deep-learning-library/ (Referenced for Model build)

https://en.wikipedia.org/wiki/Rectifier_(neural_networks)

https://en.wikipedia.org/wiki/Softmax_function

https://en.wikipedia.org/wiki/Dropout_(neural_networks)