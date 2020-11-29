import numpy as np
import cv2
import time

attente = 0

def nothing(x):
    pass

# Capture background
# Wait for 3s, blending each frame after other
def captureBG(x):
    global cap
    global background, minibackground, bgsize,y1cap,y2cap,x1cap,x2cap

    if (x == 0):
         background = None
         minibackground = None
    else: 
        ret, bg = cap.read()
        startTime = time.time()
        while( time.time() - startTime < 2) :
            ret, frame = cap.read()
            cv2.addWeighted(bg, 0.7, frame, 0.3, 0.0, bg);
        background = bg[y1cap:y2cap, x1cap:x2cap]
        minibackground = cv2.resize(bg, (bgsize, bgsize))

# Remove background using a comparison in low definition
def RemoveBackground(img, background):
    fg = cv2.GaussianBlur(img,(3,3),0)
    bg = cv2.GaussianBlur(background,(3,3),0)
  
    # We work on small areas (64x64)
    fg = cv2.resize(fg, (64, 64), interpolation = cv2.INTER_AREA)
    bg = cv2.resize(bg, (64, 64), interpolation = cv2.INTER_AREA)

    mask  = cv2.absdiff(fg, bg) # We should do this on gray images, but it works
    (B, G, R) = cv2.split(mask)

    #(Bf, Gf, Rf) = cv2.split(fg)
    #(Bb, Gb, Rb) = cv2.split(bg)
    #B = cv2.absdiff(Bf, Bb)
    #G = cv2.absdiff(Gf, Gb)
    #R = cv2.absdiff(Rf, Rb)

    # If we convert immediatly RGB to Gray, it means the Red part will have less influence on the result
    # So we will sum the 3 color channels to construct something near the distance between colors, pixel by pixel.
    # But first, to avoid overriding uint8 max value, we will truncate any value above 64 to 64
    B[B > 64] = 64
    R[R > 64] = 64
    G[G > 64] = 64
    mask = B+R+G

    # 
    th = int(np.median(mask))
    th = 36

    # Threshold total 
    ret,mask = cv2.threshold(mask,th,255,cv2.THRESH_BINARY)
    mask = cv2.resize(mask, (img.shape[0], img.shape[1]))
    cv2.imshow('FGmask',mask)
    ret,mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
    
    return cv2.bitwise_and(img,img,mask = mask)

# Use stored background image
def RemoveBackground1(img, background):

   # mask = cv2.GaussianBlur(img,(3,3),0)
   # bg = cv2.GaussianBlur(background,(3,3),0)
   # mask  = cv2.absdiff(img, background)

    # Expand a bit things
 #   kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
 #   mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Separate the channels, threshold on each
    (Bf, Gf, Rf) = cv2.split(img)
    (Bb, Gb, Rb) = cv2.split(background)
    B = cv2.absdiff(Bf, Bb)
    G = cv2.absdiff(Gf, Gb)
    R = cv2.absdiff(Rf, Rb)
    #(B, G, R) = cv2.split(mask)
    #B[B > 64] = 64
    #R[R > 64] = 64
    #G[G > 64] = 64
    mask = B+R+G
    #ret, B = cv2.threshold(B,4,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #ret, G = cv2.threshold(G,4,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #ret, R = cv2.threshold(R,4,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #mask = np.maximum(np.maximum(R, G), B)
    # mask = cv2.merge((B,G,R))
    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) 
    
    # Creates a mask
    # mask = cv2.absdiff(img, background) # Calculate difference between image and background
    # mask = cv2.addWeighted(img, 1, background, -1.0, 0.0)
    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) # Creates a mask
   
  #  gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  #  gbg = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
  #  mask = cv2.absdiff(gimg, gbg)
  #  mask = cv2.GaussianBlur(mask,(3,3),0)
  #  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
  #  mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
  
 # gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 #   gbg = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
 #   mask = cv2.absdiff(gimg, gbg)
 #   mask = cv2.GaussianBlur(mask,(3,3),0)
 #   kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
 #   mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
   
  #  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
  #  mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    cv2.imshow('FGmask',mask)

    ret,mask = cv2.threshold(mask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return cv2.bitwise_and(img,img,mask = mask)

# Remove background with a BackgroundSubtractor
def RemoveBackground2(img, background):
    global bgRemover

    mask1 = bgRemover.apply(img)
    mask = cv2.GaussianBlur(mask1,(3,3),0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    cv2.imshow('img',img)
    cv2.imshow('mask1',mask1)
    cv2.imshow('mask',mask)

    return cv2.bitwise_and(img,img,mask = mask)

# Trackbar of time before snapshot has been changed
def tb_attente_changed(x):
    global attente
    attente = x


def ComputeMaxRatio(w, h):
    m=min((h,w))
    ratio = 1
    while (m > 224*(ratio+1)) :
        ratio = ratio+1
    x1cap = (w - 224*ratio) // 2
    y1cap = (h - 224*ratio) // 2
    x2cap = x1cap + 224*ratio
    y2cap = y1cap + 224*ratio
    print("Ratio ",ratio)
    return x1cap, x2cap, y1cap, y2cap, ratio

cap = cv2.VideoCapture(1)

#bgRemover = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=None, detectShadows=False)
#bgRemover = cv2.createBackgroundSubtractorKNN(history=None, dist2Threshold=None, detectShadows=False)
#bgRemover = cv2.createBackgroundSubtractorCNT(detectShadows=False)
#bgRemover = cv2.createBackgroundSubtractorGSOC()

ratio = 0
start = time.time()
lastcap = None # lastcap
background = None # Background
minibackground = None # Thumbnail of background


# Prepare a window for the parameters
cv2.namedWindow('capture')
cv2.createTrackbar('T','capture',0,30,tb_attente_changed)
cv2.createTrackbar('Remove BG','capture',0,1,captureBG)

x1lc, x2lc, y1lc, y2lc,lcsize = (0,0,0,0,0) # Coordinates to draw the last capture
x1bg, x2bg, y1bg, y2bg, bgsize = (0,0,0,0,0) # Coordinates to draw the captured background

nbCaps = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # # Equalize Histogram
    # (B, G, R) = cv2.split(frame)
    # B = cv2.equalizeHist(B)
    # G = cv2.equalizeHist(G)
    # R = cv2.equalizeHist(R)
    # frame = cv2.merge((B,G,R))

    if (ratio == 0) :
        h,w,_ = frame.shape
        x1cap, x2cap, y1cap, y2cap, ratio = ComputeMaxRatio(w,h)
        lcsize = x1cap
        x1lc,x2lc,y1lc,y2lc=(0,lcsize,h-lcsize-1,h-1)
        bgsize=lcsize
        x1bg,x2bg,y1bg,y2bg=(0,bgsize,0,bgsize)

    currentTime = time.time()

    bRemoveBG = (cv2.getTrackbarPos('Remove BG','capture') > 0)

    # Do adjustments on image
    capture =  frame[y1cap:y2cap, x1cap:x2cap]
    capturewithbg = capture
    if (bRemoveBG):
        # Background removal
        capture = RemoveBackground(capture, background)
        frame[y1bg:y2bg, x1bg:x2bg] = minibackground
    
    # Intègre l'image modifiée
    frame = cv2.rectangle(frame, (x1cap,y1cap), (x2cap,y2cap), (255,255,0), 3)
    frame[y1cap:y2cap, x1cap:x2cap] = capture
  
     # Conserve la capture le temps est écoulé
    if ((attente > 0) and (currentTime - start > attente)) :
        capturewithbg = cv2.resize(capturewithbg, (224,224), interpolation = cv2.INTER_AREA)
        filename = "cap_%x.jpg" % (int(currentTime))  
        cv2.imwrite(filename, capturewithbg)
        if (bRemoveBG):
            capture = cv2.resize(capture, (224,224), interpolation = cv2.INTER_AREA)
            # Fun here: mutltiple capture with random BG
            filename = "cap_%x_nobg.jpg" % (int(currentTime))  
            cv2.imwrite(filename, capture)
            i=0
            while (i < 1):
                i+=1
                np.random.seed(int(currentTime) + i)
                filename = "cap_%x_randbg%x.jpg" % (int(currentTime), i)
                randombg = np.random.randint(255, size=capture.shape,dtype=np.uint8)
                mask = cv2.cvtColor(capture, cv2.COLOR_BGR2GRAY) 
                ret, mask = cv2.threshold(capture, 1, 255, cv2.THRESH_BINARY)
                capture[np.where(mask == 0)] = randombg[np.where(mask == 0)]
                cv2.imwrite(filename, capture)

        print(filename)
        lastcap = cv2.resize(capture, (lcsize,lcsize))
        nbCaps += 1
        start = currentTime
    
    if (nbCaps > 0) :
        frame[y1lc:y2lc, x1lc:x2lc] = lastcap

  
    # Display the resulting frame
    cv2.imshow('frame',frame)
 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
