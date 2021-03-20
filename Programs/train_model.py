#imports
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

#set the initial learning rate , number of epoches
#set batch size
INIT_LR= 1e-4
EPOCHS = 20
BS = 32 #batch size

DIRECTORY = r"dataset"
CATAGORIES=["mask_on","mask_off"]

# collect the list of images in the dataset
#initialize the list of images and class images 
print("Loading Images...")

data =[]
labels=[]

for category in CATAGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path= os.path.join(path,img)
        image=load_img(img_path,target_size=(224,224))#converts image to 224X224 pixels
        image=img_to_array(image)
        image=preprocess_input(image)
        #
        data.append(image)
        labels.append(category)

# perform one-hot encoding on labels
lb= LabelBinarizer()
labels=lb.fit_transform(labels)
labels=to_categorical(labels)

data = np.array(data,dtype="float32")
labels=np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)


#construct the training image generator for data augmentaion 
aug=ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)
#load the MobileNetV2 network , ensuring the head FC layer
#sets are left off
baseModel = MobileNetV2(weights="imagenet", include_top=False,
    input_tensor=Input(shape=(224, 224, 3)))

#construct the head of the model that will be placed 
# on top of the basse model
headMode1 = baseModel.output
headMode1 = AveragePooling2D(pool_size=(7,7))(headMode1)
headMode1 = Flatten(name="flatten")(headMode1)
headMode1 = Dense(128, activation="relu")(headMode1)
headMode1 = Dropout(0.5)(headMode1)
headMode1 = Dense(2, activation="softmax")(headMode1)

#Place the head FC model on top of the base model (this will 
# become the actual model we will be training )
model = Model(inputs=baseModel.input, outputs=headMode1)

#loop over all layers in the base model and freeze them
# so they will NOT be updated during the first training process
for layer in baseModel.layers:
    layer.trainable=False

#complie our model
print("Compiling Model...")
opt=Adam(lr=INIT_LR, decay=INIT_LR/EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
    metrics=["accuracy"])

#train the head of the network
print("Training Head ...")
H= model.fit(
    aug.flow(trainX,trainY , batch_size=BS),
    steps_per_epoch=len(trainX)//BS,
    validation_data=(testX, testY),
    validation_steps=len(testX)//BS,
    epochs=EPOCHS)

#make predictions on the testing set
print("Evaluating Network...")
predIdxs=model.predict(testX, batch_size=BS)

#for each image in the testing set we need to find the index of the
#label with corresponding largest predicted probability 
predIdxs= np.argmax(predIdxs, axis=1)

#Show formatted classification report 
print(classification_report(testY.argmax(axis=1),
    predIdxs,target_names=lb.classes_))

#serialize the model to the disk
print("Saving Mask detector model...")
model.save("mask_detect.model",save_format="h5")

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")