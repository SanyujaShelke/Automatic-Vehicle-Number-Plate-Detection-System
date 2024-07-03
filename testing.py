
import cv2
import numpy as np
import time

from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
import threading


license_data = ["MH12UM7096", "MH12MR5015", "MH14GU3670", "MH12PN8531", "MH19CZ7460", "MH12US3280", 
                "MH12KT1209", "MH15EW4067", "CG07CG9837", "MH48BT0957", "P334CMF", "P213GMF", "P230GMF",
                "LM56LHC", "MA08BZM", "LJ60AXF", "LP15VDV", "YT05YDP", "LB52YNV", "YC6IDLK", "LNIIXCP", 
                "MH14GH2472", "MH12OG1771"]

print("Please wait, Model is loading...")
json_file = open('./char_rec_model/MobileNets_character_recognition.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("./char_rec_model/License_character_recognition.h5")
print("Model loaded successfully...")

labels = LabelEncoder()
labels.classes_ = np.load('./char_rec_model/license_character_classes.npy')
print("Labels loaded successfully...")



#Enter path for the image to be tested
input_path = r'./test_dataset/VID_20230430_130059.mp4'

#Open the image file
cap = cv2.VideoCapture(input_path)


final_number_plate_strings = dict()

# Initialize the parameters

confThreshold = 0.4  #Confidence threshold
nmsThreshold = 0.4  #Non-maximum suppression threshold

inpWidth = 416     #Width of network's input image
inpHeight = 416     #Height of network's input image

# Load names of classes
classesFile = "yolo_utils/classes.names";

# Append all different classes into the list 'classes'
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = r"./yolo_utils/darknet-yolov3.cfg";
modelWeights = r"./yolo_utils/lapi.weights";

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]


# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom, frame):
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (0, 0, 255), cv2.FILLED)
    #cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine),    (255, 255, 255), cv.FILLED)
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        #print("out.shape : ", out.shape)
        for detection in out:
            #if detection[4]>0.001:
            scores = detection[5:]
            classId = np.argmax(scores)
            #if scores[classId]>confThreshold:
            confidence = scores[classId]
            '''if detection[4]>confThreshold:
                print(detection[4], " - ", scores[classId], " - th : ", confThreshold)
                print(detection)'''
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences.
    cropped=[]
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        
        # calculate bottom and right
        bottom = top + height
        right = left + width
        
        #crop the plate out
        cropped.append(frame[top:bottom, left:right].copy())
        # drawPred
        drawPred(classIds[i], confidences[i], left, top, right, bottom, frame)

    return cropped



# Match contours to license plate or character template
def find_contours(dimensions, img) :

    # Find all contours in the image
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Retrieve potential dimensions
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]
    
    # Check largest 5 or  15 contours for license plate or character respectively
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]

    ii = cv2.imread('contour.jpg')
    
    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs :
        # detects contour in binary image and returns the coordinates of rectangle enclosing it
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)

        # checking the dimensions of the contour to filter out the characters by contour's size
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :
        # if True:
            x_cntr_list.append(intX) #stores the x coordinate of the character's contour, to used later for indexing the contours

            char_copy = np.zeros((44,24))
            # extracting each character using the enclosing rectangle's coordinates.
            char = img[intY:intY+intHeight, intX:intX+intWidth]
            char = cv2.resize(char, (20, 40))
            
            cv2.rectangle(ii, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 2)

            # Make result formatted for classification: invert colors
            char = cv2.subtract(255, char)

            # Resize the image to 24x44 with black border
            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy) # List that stores the character's binary image (unsorted)
            # print("hi")

    # Return characters on ascending order with respect to the x-coordinate (most-left character first)
            
    # arbitrary function that stores sorted list of character indeces
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])# stores character images according to their index
        
    img_res = np.array(img_res_copy)

    return img_res


# Find characters in the resulting images
def segment_characters(image) :

    # Preprocess cropped license plate image
    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)

    img = cv2.medianBlur(img_gray_lp,5)
    img_binary_lp = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    

    # _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3,3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    # Make borders white
    img_binary_lp[0:3,:] = 255
    img_binary_lp[:,0:3] = 255
    img_binary_lp[72:75,:] = 255
    img_binary_lp[:,330:333] = 255

    # Estimations of character contours sizes of cropped license plates
    dimensions = [LP_WIDTH/6,
                       LP_WIDTH/2,
                       LP_HEIGHT/10,
                       2*LP_HEIGHT/3]

    # Get contours within cropped license plate
    char_list = find_contours(dimensions, img_binary_lp)

    return char_list



# pre-processing input images and pedict with model
def predict_from_model(image,model,labels):
    image = cv2.resize(image,(80,80))
    image = np.stack((image,)*3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    return prediction

def match_score(text1, text2):
    text1 = text1[::-1]
    text2 = text2[::-1]

    weight = [0.9, 0.9, 0.8, 0.8, 0.75, 0.75, 0.45, 0.45, 0.3, 0.3, 0.25, 0.25, 0.15, 0.15 ]
    for i in range(10):
        weight.append(1-0.1*i)
        weight.append(1-0.1*i)

    def lcs(m, n, memo):
        if (m, n) in memo:
            return memo[(m,n)]
        if m == len(text1) or n == len(text2):
            return 0
        if text1[m] == text2[n]:
            wx = 0.5 if m >= len(weight) else weight[m]
            wy = 0.5 if n >= len(weight) else weight[n]

            memo[(m,n)] = wx*wy + lcs(m+1, n+1,memo)
            return memo[(m,n)]
        else:
            memo[(m,n)] = max(lcs(m+1, n, memo), lcs(m, n+1, memo))
            return memo[(m,n)] 

    return lcs(0, 0, memo = {})
    


def match_in_db(predicted):
    score = 0
    res = ""
    for real in license_data:
        curr_score = match_score(predicted, real)
        if curr_score > score:
            score = curr_score
            res = real
    return (res, score)


def extract_license_number(tmpFrame, frame_no, timestamp):
    print("frame no: ", frame_no)
    if frame_no < 0:
        return 
    
    frame = tmpFrame.copy()

    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    def extract_parallel(outs, frame_no, timestamp):
        print("started parallely")
        # Remove the bounding boxes with low confidence
        cropped = postprocess(frame, outs)

        charArr = []
        for cropped_img in cropped:
            charArr.append(segment_characters(cropped_img))
            cv2.imwrite("./test/" + str(frame_no)+".jpg", cropped_img)

        res = []
        for char in charArr:
            final_string = ''
            for i,character in enumerate(char):
                title = np.array2string(predict_from_model(character,model,labels))
                final_string+=title.strip("'[]")
            if len(final_string) > 1:
                res.append(final_string)

        final_matched_no_plates = []
        for predicted in res:
            matched_plate, matched_score = match_in_db(predicted)
            final_matched_no_plates.append(matched_plate)

        if len(final_matched_no_plates):
            final_number_plate_strings[frame_no] = [timestamp, final_matched_no_plates, matched_score, res]

        # if len(res):
        #     final_number_plate_strings[frame_no] = [timestamp, res]
    
    threading.Thread(target=extract_parallel, args=(outs, frame_no, 0,)).start()



curr_thread = threading.Thread(target=extract_license_number, args=([], -1, 0,))
frame_no = 0
# Create a window to display the video
cv2.namedWindow('video', cv2.WINDOW_NORMAL)
while cap.isOpened():
    hasFrame, frame = cap.read() #frame: an image object from cv2
    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        break

    frame_no = frame_no + 1
    if not curr_thread.is_alive():
        curr_thread = threading.Thread(target=extract_license_number, args=(frame, frame_no, 0,))
        curr_thread.start()
        
    time.sleep(1/cap.get(cv2.CAP_PROP_FPS))
    cv2.imshow('video', frame)

    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

    print(final_number_plate_strings)
