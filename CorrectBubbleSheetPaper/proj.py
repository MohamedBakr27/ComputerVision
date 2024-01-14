import cv2 as cv
import numpy as np
import imutils
from imutils import contours
import os

# Author : Mohamed Bakr

def getThresh(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    edged = cv.Canny(blur, 100, 200)
    return cv.threshold(edged, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

# Read Correct Answers from Answer Module.

answer_module_path = input('Enter path of answer module : ')
to_correct_path = input('Enter path of papers to correct : ')

os.chdir(answer_module_path)

answer_module = cv.imread('answer_module.png')
answers = {}

answers_module_thresh = getThresh(answer_module)
cnts = imutils.grab_contours(cv.findContours(answers_module_thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE))
answer_module = cv.GaussianBlur(answer_module, (5, 5), 0)

qcnts = []
for cnt in reversed(cnts):
    (x,y,w,h) = cv.boundingRect(cnt)
    aspect = w/float(h)
    if w>=20 and h>=20 and aspect >= 0.9 and aspect <= 1.1:
        qcnts.append(cnt)

for (q,i) in enumerate(np.arange(0,len(qcnts),4)):
    cnts = contours.sort_contours(qcnts[i:i+4])[0]
    for (j,c) in enumerate(cnts):
        (x,y,w,h) = cv.boundingRect(c)
        current_cnt = cv.rectangle(answer_module.copy(),(x,y),(x+w,y+h),(0,255,0),thickness=2)
        mask = np.zeros(answers_module_thresh.shape,dtype='uint8')
        cv.drawContours(mask,[c],-1,255,-1)
        mask = cv.bitwise_and(answers_module_thresh,answers_module_thresh,mask=mask)
        if cv.countNonZero(mask)<400:
            answers[i//4] = j
            break

# Mark of Exam Papers.

os.chdir(to_correct_path)

papers = os.listdir()
results = []
for (no,paper) in enumerate(papers):
    img = cv.imread(paper)
    img_thresh = getThresh(img)
    img = cv.GaussianBlur(img, (5, 5), 0)
    cnts = imutils.grab_contours(cv.findContours(img_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE))

    qcnts = []
    for cnt in reversed(cnts):
        (x, y, w, h) = cv.boundingRect(cnt)
        aspect = w / float(h)
        if w >= 20 and h >= 20 and aspect >= 0.9 and aspect <= 1.1:
            qcnts.append(cnt)

    correct = 0
    marked_img = cv.putText(img.copy(), f'Correct = {correct}', (15, 33), cv.FONT_HERSHEY_TRIPLEX, 1.0, (235, 183, 0),thickness=2)
    cv.imshow('Current Paper', marked_img)

    cv.waitKey(50)

    for (q, i) in enumerate(np.arange(0, len(qcnts), 4)):
        cnts = contours.sort_contours(qcnts[i:i + 4])[0]
        marked = 0
        correct_c = False
        for (j, c) in enumerate(cnts):
            (x, y, w, h) = cv.boundingRect(c)
            current_cnt = cv.circle(marked_img.copy(), (x+(w//2), y+(h//2)), 45, (255, 0, 0), thickness=4)
            mask = np.zeros(img_thresh.shape, dtype='uint8')
            cv.drawContours(mask, [c], -1, 255, -1)
            mask = cv.bitwise_and(img_thresh, img_thresh, mask=mask)
            cv.imshow('Current Paper', current_cnt)

            if cv.countNonZero(mask) < 400:
                (x, y, w, h) = cv.boundingRect(c)
                if answers[i//4] == j:
                    correct_c = True
                cv.circle(img, (x + (w // 2), y + (h // 2)), 45, (0, 255, 0) if answers[i//4]==j else (0,0,255), thickness=-1)
                marked_img = cv.putText(img.copy(), f'Correct = {correct}', (15, 33), cv.FONT_HERSHEY_TRIPLEX, 1.0,(235, 183, 0), thickness=2)
                cv.imshow('Current Paper', marked_img)
                cv.waitKey(100)
                correct += 1 if answers[i//4]==j else 0
                marked += 1
            cv.waitKey(100)

        if marked>1 and correct_c:
            correct -= 1

        marked_img = cv.putText(img.copy(), f'Correct = {correct}', (15, 33), cv.FONT_HERSHEY_TRIPLEX, 1.0,(235, 183, 0), thickness=2)
        cv.imshow('Current Paper', marked_img)
        cv.waitKey(100)

    cv.imshow('Current Paper', marked_img)
    cv.waitKey(200)
    cv.imwrite(f'{no+1}c.png', marked_img)
    results.append((no,correct))
    if cv.waitKey(5000) & 0xFF == ord('\n'):
        continue

print(results)