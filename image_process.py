import numpy as np
from imutils.contours import sort_contours
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import math

# loading pre trained model
model = tf.keras.models.load_model('model-a.keras')
model_diacritic = tf.keras.models.load_model('model-diac.keras')
image_shape = 50

classnames = ['a', 'ba', 'dara', 'ei', 'ga', 'ha', 'ka', 'la', 'ma', 'na', 'nga', 'ou', 'pa', 'sa', 'ta', 'wa', 'ya']
classnames_diacritic = ['bar', 'plus', 'dots', 'x']

def process(path):
    img = cv2.imread(path)
    img_org = img

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Converts to grayscale
    ret, thresh = cv2.threshold(img, 127, 225, cv2.THRESH_BINARY) # Converts to binary
    # Seperate characters
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    bounds = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if (w >= 25 and w <= 500) and (h >= 50 and h <= 500) and (bb_intersection_over_union((x,y,x+w,y+h), list(pos[1] for pos in bounds)) < 0.5):
            # putting boundary on each digit
            cv2.rectangle(img_org,(x,y),(x+w,y+h),(0,255,0),2)
            # crop image
            cropped = img[y:y+h, x:x+w]
            cropped = image_refiner(cropped, constant_values=(255,), org_size = (image_shape-(image_shape//4)), img_size = image_shape)
            # apply threshold
            th, fnl = cv2.threshold(cropped, 127, 255, cv2.THRESH_BINARY)
            # apply morphology
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
            morph = cv2.morphologyEx(fnl, cv2.MORPH_CLOSE, kernel)
            
            baybayin = predict(morph)
            bounds.append([baybayin, (x,y,x+w,y+h)])
            img_org = put_label(img_org,baybayin,x,y)    

    diacritics = detect_diacritic(path)

    for diacritic, diacritic_bound in diacritics:
        (x,y,w,h) = diacritic_bound
        if bb_intersection_over_union((x,y,x+w,y+h), list(pos[1] for pos in bounds)) < 0.85:
            cv2.rectangle(img_org,(x,y),(x+w,y+h),(0,0,255),2)
            # img_org = put_label(img_org,diacritic,x,y)
    
    syllables, labels = detect_syllables(bounds, diacritics)
    if (len(syllables) > 0):
        for i, syllable in enumerate(syllables):
            (x,y,w,h) = syllable
            cv2.rectangle(img_org,(x,y),(w,h),(255,0,0),2)
            img_org = put_label(img_org,labels[i],max(x,w),max(y,h))

    cv2.imwrite('output.jpg', img_org)
    return img_org

def detect_diacritic(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Converts to grayscale
    ret, thresh = cv2.threshold(img, 127, 225, cv2.THRESH_BINARY) # Converts to binary
    # Seperate characters
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    diacritics = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if (w >= 2 and w < 50) and (h >= 2 and h < 50): #and (bb_intersection_over_union((x,y,x+w,y+h), (list(bounds.values()))) < 0.75):
            # crop image
            cropped = img[y:y+h, x:x+w]
            # apply threshold
            th, fnl = cv2.threshold(cropped, 127, 255, cv2.THRESH_BINARY_INV)
            fnl = image_refiner(fnl)
            diacritic = predict_diacritic(fnl)
            diacritics.append([diacritic, (x,y,w,h)])      
    return diacritics 

def predict(_img):
    img = _img.reshape(-1,image_shape,image_shape,1)
    return classnames[np.argmax(model.predict(img))]

def predict_diacritic(_img):
    img = _img.reshape(-1,50,50,1)
    return classnames_diacritic[np.argmax(model_diacritic.predict(img))]

def detect_syllables(letters, diacritics): 
    if (len(diacritics) <= 0):
        return [], []   
    
    syllable = []
    label = []
    for letter_label, (x,y,w,h) in letters:
        diacritics_score = [math.inf]
        current_syllable = ()
        current_label = ""
        (x,y,w,h) = (x,y,w-x,h-y) 
        p = math.ceil(math.sqrt(h*w))
        for diacritics_label, (x2,y2,w2,h2) in diacritics:
            (a,b) = center_bb((x,y,w,h))         
            (c,d) = center_bb((x2,y2,w2,h2))   
            if (c >= x and c <= (x+w)) and (d >= (y-p) and d <= y):
                # top
                curr = distance(a,b,c,d)
                if(min(diacritics_score) > curr):
                    current_syllable = (x,y2,(x+w),(y+h))
                    current_label = compute_syllable(letter_label, diacritics_label, "top")
                diacritics_score.append(curr)
            if (c >= x and c <= x+w) and (d >= y+h and d <= y+h+p):
                # bottom
                curr = distance(a,b,c,d)
                if(min(diacritics_score) > curr):
                    current_syllable = (x,y,(x+w),(y2+h2))
                    current_label = compute_syllable(letter_label, diacritics_label, "bot")
                diacritics_score.append(curr)
        if (len(current_syllable) > 0):
            syllable.append(current_syllable)
        if (current_label != ""):
            print(current_label)
            label.append(current_label)
    return syllable, label
  
def compute_syllable(baybayin, diacritic, pos):
    classnames_plus = ['a', 'b', 'd', 'ei', 'g', 'h', 'k', 'l', 'm', 'n', 'ng', 'ou', 'p', 's', 't', 'w', 'y']
    classnames_dot_top = ['a', 'be', 'de', 'ei', 'ge', 'he', 'ke', 'le', 'me', 'ne', 'nge', 'ou', 'pe', 'se', 'te', 'we', 'ye']
    classnames_dot_bot = ['a', 'bo', 'do', 'ei', 'go', 'ho', 'ko', 'lo', 'mo', 'no', 'ngo', 'ou', 'po', 'so', 'to', 'wo', 'yo']
    classnames_bar_top = ['a', 'bi', 'di', 'ei', 'gi', 'hi', 'ki', 'li', 'mi', 'ni', 'ngi', 'ou', 'pi', 'si', 'ti', 'wi', 'yi']
    classnames_bar_bot = ['a', 'bu', 'du', 'ei', 'gu', 'hu', 'ku', 'lu', 'mu', 'nu', 'ngu', 'ou', 'pu', 'su', 'tu', 'wu', 'yu']
    match(diacritic):
        case 'bar':
            if(pos == "top"):
                return classnames_bar_top[classnames.index(baybayin)]
            elif(pos == "bot"):
                return classnames_bar_bot[classnames.index(baybayin)]
        case 'plus':
            if(pos == "bot"):
                return classnames_plus[classnames.index(baybayin)]
            elif(pos == "top"):
                return baybayin
        case 'dots':
            if(pos == "top"):
                return classnames_dot_top[classnames.index(baybayin)]
            elif(pos == "bot"):
                return classnames_dot_bot[classnames.index(baybayin)]
        case 'x':
            if(pos == "bot"):
                return classnames_plus[classnames.index(baybayin)]
            elif(pos == "top"):
                    return baybayin
    return None
        
def put_label(t_img,label,x,y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    l_x = int(x) - 10
    l_y = int(y) + 10
    cv2.rectangle(t_img,(l_x,l_y+5),(l_x+35,l_y-35),(0,255,0),-1) 
    cv2.putText(t_img,str(label),(l_x,l_y), font,1.5,(255,0,0),1,cv2.LINE_AA)
    return t_img
def image_refiner(img, constant_values = (0,), org_size = 45, img_size = 50):
    rows,cols = img.shape
    if rows > cols:
        factor = org_size/rows
        rows = org_size
        cols = int(round(cols*factor))
    else:
        factor = org_size/cols
        cols = org_size
        rows = int(round(rows*factor))
    img = cv2.resize(img, (cols, rows))
    
    #get padding 
    colsPadding = (int(math.ceil((img_size-cols)/2.0)),int(math.floor((img_size-cols)/2.0)))
    rowsPadding = (int(math.ceil((img_size-rows)/2.0)),int(math.floor((img_size-rows)/2.0)))
    
    img = np.lib.pad(img,(rowsPadding,colsPadding),'constant', constant_values=constant_values)
    return img

def bb_intersection_over_union(boxA, _boxB):
    ious = [0]
    
    for boxB in _boxB:
	    # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        ious.append(iou)
	# return the intersection over union value
    return max(ious)

def center_bb(box):
    (x,y,w,h) = box
    x = (x + (x + w)) / 2
    y = (y + (y + h)) / 2
    return (x,y)

def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def show_img(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
