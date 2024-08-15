import numpy as np
import cv2
import tensorflow as tf
import math
import sys
# loading pre trained model
model = tf.keras.models.load_model('models/model-baybayin.keras')
model_diacritic = tf.keras.models.load_model('models/model-diac.keras')

classnames = ['a', 'ba', 'da', 'e', 'ga', 'ha', 'ka', 'la', 'ma', 'na', 'nga', 'o', 'pa', 'sa', 'ta', 'wa', 'ya']
classnames_diacritic = ['bar', 'plus', 'dots', 'x']

class Syllable:
    def __init__(self, position, label, distance, pair):
        self.position = position #(x,y,x+w,y+h)
        self.label = label
        self.distance = distance # distance between baybayin and diacritics
        self.pair = pair
    def is_same(self, syllable_b):
        _, diacritic_a = self.pair
        _, diacritic_b = syllable_b.pair
        return diacritic_a == diacritic_b
    def revert(self):
        self.position = self.pair[0]
        if "ng" in self.label:
            self.label = "nga"
        else:
            self.label = self.label[0] + "a"
        self.pair = (self.pair[0], None)

def process(path):
    img = cv2.imread(path)
    img_org = img

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Converts to grayscale
    ret, thresh = cv2.threshold(img, 127, 225, cv2.THRESH_BINARY) # Converts to binary
    # Seperate characters
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)

    # Detect Baybayin
    bounds = []
    for j, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if (hierarchy[0][j][3]!=-1) and (w >= 25 and w <= 500) and (h >= 50 and h <= 500) and (bb_intersection_over_union((x,y,x+w,y+h), list(pos[1] for pos in bounds)) < 0.5):
            cv2.rectangle(img_org,(x,y),(x+w,y+h),(0,255,0),2) # putting boundary on each digit
            # crop image
            cropped = img[y:y+h, x:x+w]
            cropped = image_refiner(cropped, constant_values=(255,), org_size = 45, img_size = 50) # CHANGE IMG SIZE AND ORG SIZE IF THE IMG SHAPE IN THE MODEL IS DIFFERENT
            # apply threshold
            th, fnl = cv2.threshold(cropped, 127, 255, cv2.THRESH_BINARY)
            # apply morphology
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
            morph = cv2.morphologyEx(fnl, cv2.MORPH_CLOSE, kernel)
            baybayin = predict(morph)
            bounds.append([baybayin, (x,y,x+w,y+h)])
            # img_org = put_label(img_org,baybayin,x,y)    

    # Detect diacritics
    diacritics = detect_diacritic(path, bounds)
    for diacritic, diacritic_bound in diacritics:
        (x,y,w,h) = diacritic_bound
        cv2.rectangle(img_org,(x,y),(x+w,y+h),(0,0,255),2)
        # img_org = put_label(img_org,diacritic,x,y)

    syllables = detect_syllables(bounds, diacritics)
    for syllable in syllables:
        print(syllable)
        (x,y,w,h) = syllable.position
        labels = syllable.label
        cv2.rectangle(img_org,(x,y),(w,h),(255,0,0),2)
        img_org = put_label(img_org,labels,max(x,w),max(y,h))

    cv2.imwrite('static/temp/output.jpg', img_org)
    return img_org
def detect_diacritic(path, baybayin):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Converts to grayscale
    ret, thresh = cv2.threshold(img, 127, 225, cv2.THRESH_BINARY) # Converts to binary
    # Seperate characters
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    diacritics = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        inside = False
        for b, b_pos in baybayin:
            if is_inside(b_pos, (x,y,w,h)) == True:
                inside = True
        if (w >= 2 and w < 50) and (h >= 2 and h < 50) and (inside == False):
            # crop image
            cropped = img[y:y+h, x:x+w]
            # apply threshold
            th, fnl = cv2.threshold(cropped, 127, 255, cv2.THRESH_BINARY_INV)
            fnl = image_refiner(fnl)
            diacritic = predict_diacritic(fnl)
            diacritics.append([diacritic, (x,y,w,h)])    

    return diacritics 
def is_inside(box1, box2):
    x, y, w, h = box1
    x2, y2, w2, h2 = box2
    if x <= x2 <= w and y <= y2 <= h and x2+w2 <= w and y2+h2 <= h:
        return True
    return False
def predict(_img):
    img = _img.reshape(-1,50,50,1) # CHANGE TO (-1, new_x, new_y, 1) IF THE IMG SHAPE IN THE MODEL IS DIFFERENT
    return classnames[np.argmax(model.predict(img))]
def predict_diacritic(_img):
    img = _img.reshape(-1,50,50,1)
    return classnames_diacritic[np.argmax(model_diacritic.predict(img))]
def detect_syllables(letters, diacritics): 
    syllable = [] # [(x,y,w,h), label, dist, (baybayin_pos, diacritics_pos)]
    for letter_label, (x,y,w,h) in letters:
        diacritics_score = math.inf
        current_position = ()
        current_label = ""
        (x,y,w,h) = (x,y,w-x,h-y)
        diacritics_pos = (0,0,0,0) # (x2,y2,w2,h2)
        p = math.ceil(math.sqrt(h*w))
        for diacritics_label, (x2,y2,w2,h2) in diacritics:
            (a,b) = center_bb((x,y,w,h))         
            (c,d) = center_bb((x2,y2,w2,h2))   
            curr = distance(a,b,c,d)
            if (c >= x and c <= (x+w)) and (d >= (y-p) and d <= y) and (diacritics_score > curr):
                # top
                current_position = (x,y2,(x+w),(y+h))
                current_label = compute_syllable(letter_label, diacritics_label, "top")
                diacritics_score = curr
                diacritics_pos = (x2,y2,w2,h2)
            elif (c >= x and c <= x+w) and (d >= y+h and d <= y+h+p) and (diacritics_score > curr):
                # bottom
                current_position = (x,y,(x+w),(y2+h2))
                current_label = compute_syllable(letter_label, diacritics_label, "bot")
                diacritics_score = curr
                diacritics_pos = (x2,y2,w2,h2)
 
        if (len(current_position) > 0) and (current_label != ""):
            syllable.append(Syllable(current_position, current_label, diacritics_score, ((x,y,x+w,y+h), diacritics_pos)))
        else:
            syllable.append(Syllable((x,y,x+w,y+h), letter_label, 0, ((x,y,x+w,y+h), diacritics_pos)))
    
    # Resolve conflicts
    for i in range(len(syllable)):
       for j in range(i+1, len(syllable)):
              if syllable[i].is_same(syllable[j]):
                     if syllable[i].distance > syllable[j].distance:
                            syllable[i].revert()
                     else:
                            syllable[j].revert()
    return syllable

def compute_syllable(baybayin, diacritic, pos):
    classnames_plus =    ['a', 'b', 'd', 'e', 'g', 'h', 'k', 'l', 'm', 'n', 'ng', 'o', 'p', 's', 't', 'w', 'y']
    classnames_dot_top = ['a', 'be', 'de', 'e', 'ge', 'he', 'ke', 'le', 'me', 'ne', 'nge', 'o', 'pe', 'se', 'te', 'we', 'ye']
    classnames_dot_bot = ['a', 'bo', 'do', 'e', 'go', 'ho', 'ko', 'lo', 'mo', 'no', 'ngo', 'o', 'po', 'so', 'to', 'wo', 'yo']
    classnames_bar_top = ['a', 'bi', 'di', 'e', 'gi', 'hi', 'ki', 'li', 'mi', 'ni', 'ngi', 'o', 'pi', 'si', 'ti', 'wi', 'yi']
    classnames_bar_bot = ['a', 'bu', 'du', 'e', 'gu', 'hu', 'ku', 'lu', 'mu', 'nu', 'ngu', 'o', 'pu', 'su', 'tu', 'wu', 'yu']
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

def __main__():
    input = sys.argv[1]
    output = process(input)
    sys.stdout.flush()