
import cv2
import numpy as np
import time

from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
import threading
from datetime import datetime
import time
from utils import *

license_data = ["MH12UM7096", "MH12MR5015", "MH14GU3670", "MH12PN8531", "MH19CZ7460", "MH12US3280", 
                "MH12KT1209", "MH15EW4067", "CG07CG9837", "MH48BT0957", "P334CMF", "P213GMF", "P230GMF",
                "LM56LHC", "MA08BZM", "LJ60AXF", "LP15VDV", "YT05YDP", "LB52YNV", "YC6IDLK", "LNIIXCP", 
                "MH14GH2472", "MH12OG1771", "MH12UC1253", "UP32MD7044", "MH12SY0369", "MH12UC3807", 
                "MH12UN2824", "MH12GM6007", "MH14KO2243", "MH14KN7801", "MH14EG0929", "MH12UM7069", 
                "MH12MR5015", "ICNK46", "1WGE75", "1CE249", "2HDP37", "1RDR25", "1KXP93", "AAN6530"]


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

final_number_plate_strings = dict()


confThreshold = 0.4 
nmsThreshold = 0.4
MATCH_SCORE_THRESHOLD = 2.0
MSG_TIME_INTERVAL = 15000 # in miliseconds
FREQ_TIME_THRESHOLD = 7000  

inpWidth = 416
inpHeight = 416

# load class names
classesFile = "yolo_utils/classes.names";

classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

modelConfiguration = r"./yolo_utils/darknet-yolov3.cfg";
modelWeights = r"./yolo_utils/lapi.weights";

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]


# Draw bounding box
def drawPred(classId, conf, left, top, right, bottom, frame):
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

    label = '%.2f' % conf

    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    # display label and bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (0, 0, 255), cv2.FILLED)
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)

# use non maximum suppresion to remove low confidence bounding boxes
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

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

    cropped=[]
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold) # Non maximum suppresion 
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        
        bottom = top + height
        right = left + width
        
        # croping the plate out
        cropped.append(frame[top:bottom, left:right].copy())
        drawPred(classIds[i], confidences[i], left, top, right, bottom, frame)

    return cropped


# Find the conotours of the cropped license plate image
def find_contours(dimensions, img) :
    # get all contours
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]
    
    # Check largest 15 contours
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]

    # ii = cv2.imread('contour.jpg')
    
    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs :
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)

        # reject the character which doesn't follow dimentions
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :
            x_cntr_list.append(intX) 

            char_copy = np.zeros((44,24))
            char = img[intY:intY+intHeight, intX:intX+intWidth]
            char = cv2.resize(char, (20, 40))
            
            # cv2.rectangle(ii, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 2)

            char = cv2.subtract(255, char)

            # Resize the image to 24x44 with black border
            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy)

    # Return characters sequentially sorted according to x co-ordinate
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])
        
    img_res = np.array(img_res_copy)
    return img_res


# segment the characters from the cropped image
def segment_characters(image) :
    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)

    img = cv2.medianBlur(img_gray_lp,5)
    img_binary_lp = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    
    img_binary_lp = cv2.erode(img_binary_lp, (3,3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

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


def predict_from_model(image,model,labels):
    image = cv2.resize(image,(80,80))
    image = np.stack((image,)*3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    return prediction

# calculate the matching score of predicted no plate and actual no plate from db
def match_score(text1, text2):
    text1 = text1[::-1]
    text2 = text2[::-1]

    # weights to be used during LCS, as last characters of no plate has more importance than first
    weight = [0.9, 0.9, 0.8, 0.8, 0.75, 0.75, 0.45, 0.45, 0.3, 0.3, 0.25, 0.25, 0.15, 0.15 ]
    for i in range(10):
        weight.append(1-0.1*i)
        weight.append(1-0.1*i)

    # longest common subsequence
    def lcs(m, n, memo):
        if (m, n) in memo:
            return memo[(m,n)]
        if m == len(text1) or n == len(text2):
            return 0
        if text1[m] == text2[n]:
            wx = 0.5 if m >= len(weight) else weight[m]
            wy = 0.5 if n >= len(weight) else weight[n]

            # mutiply by weight of respective indices
            memo[(m,n)] = wx*wy + lcs(m+1, n+1,memo)
            return memo[(m,n)]
        else:
            memo[(m,n)] = max(lcs(m+1, n, memo), lcs(m, n+1, memo))
            return memo[(m,n)] 

    return lcs(0, 0, memo = {})
    

# match the predicted plate in the database
def match_in_db(predicted):
    score = 0
    res = ""
    for real in license_data:
        # calculate match score
        curr_score = match_score(predicted, real)
        if curr_score > score:
            score = curr_score
            res = real
    # return plate which is matched most according to match score
    return (res, score)


# send message to vehicle owner
def send_msg(frame, frame_no, cropped_frame, license_no, timestamp, score, final_string):
    _, encoded_frame = cv2.imencode(".jpg", frame)
    _, encoded_cropped_frame = cv2.imencode(".jpg", cropped_frame)

    send_mail("yerkalsm19.comp@coep.ac.in", [encoded_frame, encoded_cropped_frame], license_no, timestamp)
    print("MSG Sent : ", frame_no, score, license_no, final_string)


threadArr = []

def extract_license_number(tmpFrame, frame_no, timestamp):
    if frame_no < 0:
        return 
    frame = tmpFrame.copy()

    blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
    net.setInput(blob)

    outs = net.forward(getOutputsNames(net))

    def extract_parallel(outs, frame, frame_no, timestamp):
        cropped = postprocess(frame, outs)

        for cropped_img in cropped:
            char = segment_characters(cropped_img)
            cv2.imwrite("./test/" + str(frame_no)+".jpg", cropped_img)

            final_string = ''
            for i,character in enumerate(char):
                title = np.array2string(predict_from_model(character,model,labels))
                final_string += title.strip("'[]")

            if len(final_string) > 2:
                matched_plate, matched_score = match_in_db(final_string)
                if matched_score > MATCH_SCORE_THRESHOLD:
                    recent_time = final_number_plate_strings.get(matched_plate, -1)
                    final_number_plate_strings[matched_plate] = timestamp
                    if recent_time == -1 or timestamp - recent_time > MSG_TIME_INTERVAL:
                        send_msg(frame, frame_no, cropped_img, matched_plate, timestamp, matched_score, final_string)  

                elif matched_score > 1.3: 
                    recent_time = final_number_plate_strings.get(matched_plate, -1)
                    final_number_plate_strings[matched_plate] = timestamp
                    if recent_time != -1 and timestamp - recent_time > FREQ_TIME_THRESHOLD:
                        print("Double frame capture")
                        send_msg(frame, frame_no, cropped_img, matched_plate, timestamp, matched_score, final_string)

        # if len(final_matched_no_plates):
        #     final_number_plate_strings[frame_no] = [timestamp, final_matched_no_plates, matched_score, res]

        # if len(res):
        #     final_number_plate_strings[frame_no] = [timestamp, res]
    
    threadArr.append(threading.Thread(target=extract_parallel, args=(outs, frame, frame_no, timestamp,)))
    threadArr[-1].start()



def start_video_surveillance(input_path):
    # input video path
    # input_path = r'./test_dataset/trim.mp4'
    cap = cv2.VideoCapture(input_path)

    curr_thread = threading.Thread(target=extract_license_number, args=([], -1, 0,))
    frame_no = 0
    # Create a window to display the video
    cv2.namedWindow('video', cv2.WINDOW_NORMAL)
    while cap.isOpened():
        hasFrame, frame = cap.read() #frame: an image object from cv2
        timestamp = int(round(time.time() * 1000))

        # now = datetime.now()
        # dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

        if not hasFrame:
            print("Done processing !!!")
            break

        frame_no = frame_no + 1
        if not curr_thread.is_alive():
            curr_thread = threading.Thread(target=extract_license_number, args=(frame, frame_no, timestamp,))
            curr_thread.start()
            
        time.sleep(1/cap.get(cv2.CAP_PROP_FPS))
        cv2.imshow('video', frame)

        if cv2.waitKey(1) & 0xFF == ord('e'):
            break


    for th in threadArr:
        th.join()
