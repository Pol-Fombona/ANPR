import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils
import re
import os
from imutils import contours
from os import listdir
import easyocr
import re



def read_image(filename):
    # Read Image in Gray Scale
    return cv2.imread(filename, 0)

def get_location(image):
    keypoints = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    num_contours = 10 if len(contours) > 10 else len(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:num_contours]

    location = None
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        approx = cv2.boxPoints(rect)
        approx = np.int0(approx)
        if len(approx) == 4:
            points = np.squeeze(approx)

            X, Y = points[:,0], points[:,1]

            left_y, right_y = np.min(Y), np.max(Y)
            top_x, bottom_x = np.min(X), np.max(X)

            x, y = right_y - left_y, bottom_x - top_x
            ratio = y / x

            if (DEBUG): print(f'\nRatio: {ratio} ({y}/{x})\n')

            if 8 > ratio > 1.8:
                location = approx
                break

    return location

def show_image(img, title = ''):

    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def metode_B(image, og_img):

    history = []

    blurred = cv2.GaussianBlur(image, (5,5), 0) 
    history.append((blurred, "Gaussian Blur"))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    blackHat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel, iterations=3)
    history.append((blackHat, "Black Hat"))

    treshold = cv2.threshold(blackHat, blackHat.max()//2, 255, cv2.THRESH_BINARY)[1]
    #treshold = cv2.threshold(blackHat, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]    
    history.append((treshold, "Treshold"))
  
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,1))
    dilate = cv2.dilate(treshold, horizontal_kernel, iterations=7)
    history.append((dilate, "Dilate"))

    opening = cv2.morphologyEx(dilate, cv2.MORPH_OPEN, kernel, iterations=1)
    history.append((opening, "Opening"))

    gaussianBlur = cv2.GaussianBlur(opening, (3,3), 0)
    history.append((gaussianBlur, "Second Gaussian"))

    treshold2 = cv2.threshold(gaussianBlur, gaussianBlur.max()//2, 255, cv2.THRESH_BINARY)[1]
    history.append((treshold2, "Second Treshold"))



    square_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    closing = cv2.morphologyEx(treshold2, cv2.MORPH_CLOSE, square_kernel, iterations=5)
    history.append((closing, "Closing"))

    

    # Filter using contour area and remove small noise
    cnts = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 1000:
            cv2.drawContours(closing, [c], -1, (0,0,0), -1)

    history.append((closing, "Contours"))

    location = get_location(closing.copy())
    
    if location is None:
        if (DEBUG): print('Location not found!')
        return 


    mask = np.zeros(og_image.shape, np.uint8)
    y, x = og_image.shape
    for i in range(4):
        location[i][0] = location[i][0] / (720/x)
        location[i][1] = location[i][1] / (720/x)

 
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(og_image, og_image, mask=mask)

    warped = homography(new_image,location)
    history.append((warped, "Warped"))

    ret2,th2 = cv2.threshold(warped, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th2 = removeEUStrip(th2, 120)

    if (DEBUG):
        for item in history:
            show_image(item[0], item[1])


    return th2


"""
def imgToText(img, reader):
    result  = reader.readtext(img, detail=0, allowlist = '0123456789BCDFGHJKLMNPRSTVWXYZ')

    if len(result) == 0:
        return ''
    
    elif len(result) >= 1:
        result = ''.join(result)

    
    return result.replace(" ", "")
"""

def imgToText(img, reader):
    result  = reader.readtext(img, detail=1, allowlist = '0123456789BCDFGHJKLMNPRSTVWXYZ')

    if len(result) == 0:
        return ''

    elif type(result[0]) is tuple:
        if len(result) == 1:
            itemp = img.copy()

            x = result[0][0][0][0]
            y = result[0][0][0][1]
            w = result[0][0][1][0] - x
            h = result[0][0][2][1] - y
            cv2.rectangle(itemp, (x, y), (x+w, y+h), (0, 0, 255), 1)
            cv2.imshow('', itemp)

            result = '' + result[0][1]

        else:
            itemp = img.copy()

            x1 = result[0][0][0][0]
            y1 = result[0][0][0][1]
            w1 = result[0][0][1][0] - x1
            h1 = result[0][0][2][1] - y1

            x2 = result[1][0][0][0]
            y2 = result[1][0][0][1]
            w2 = result[1][0][1][0] - x2
            h2 = result[1][0][2][1] - y2
            cv2.rectangle(itemp, (x1, y1), (x1+w1, y1+h1), (0, 0, 255), 1)
            cv2.rectangle(itemp, (x2, y2), (x2+w2, y2+h2), (0, 0, 255), 1)
            cv2.imshow('', itemp)
            #cv2.waitKey(0)
            temp = result[0][1] + result[1][1]
            result = '' + temp


    else:
        result = ''.join(result)


    return result.replace(" ", "")

def orderPoints(pts):
	rect = np.zeros((4, 2), dtype = "float32")

	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect

def homography(gray_img,location):
	src = orderPoints(np.squeeze(location).astype(np.float32))
	(tl, tr, br, bl) = src
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	M = cv2.getPerspectiveTransform(src, dst) 
	warped = cv2.warpPerspective(gray_img, M, (maxWidth, maxHeight))
	return warped

def removeEUStrip(img, thr):
  mean_columns = img.mean(0)

  for ind, val in enumerate(mean_columns):
    if val > thr:
      break

  img_ret = np.delete(img, [i for i in range(ind)], axis=1)

  return img_ret

def checkText(text):

    if len(text) > 7:
        split_text = (re.split('(\d+)',text))

        numbers = split_text[1]
        text = split_text[2]

        if len(numbers) > 4:
            numbers = numbers[1:]

        if len(text) > 3:
            text = text[:3]


        text = numbers + text

    return text


if __name__ == '__main__':

    directory = os.getcwd()
    
    DIR_IMAGES = directory + "\\images\\"
    #DIR_IMAGES = directory + "\\test\\"
    DIR_CORRECT_RESULTS = directory + "\licensePlate\correct\\"
    DIR_INCORRECT_RESULTS = directory + "\licensePlate\incorrect\\"

    reader = easyocr.Reader(['en'])

    DEBUG = False
    total = 0

    files = listdir(DIR_IMAGES)

    for filename in files:

        img = read_image(DIR_IMAGES + filename)   
        og_image = img.copy()
        
        y, x = img.shape
        # Resize Image
        r = 720 / x
        dim = (720, int(y * r))
        img = cv2.resize(img, dim)

        plateLocation = metode_B(img, og_image)


        if "_" in filename:
            filePlateTag = filename[:filename.rindex('_')]
        else:
            filePlateTag = filename[:filename.rindex('.')]

        if (plateLocation is None):
            print("File:", filename, "- License Plate not found")          

        else:
            

            licensePlateNumber = imgToText(plateLocation, reader)

            if licensePlateNumber == filePlateTag:
                cv2.imwrite(DIR_CORRECT_RESULTS + filename, plateLocation) 
                total += 1

            else:
                temp = licensePlateNumber
                licensePlateNumber = checkText(licensePlateNumber)

                if licensePlateNumber == filePlateTag:
                    cv2.imwrite(DIR_CORRECT_RESULTS + filename, plateLocation) 
                    total += 1
                else:
                    cv2.imwrite(DIR_INCORRECT_RESULTS + filename, plateLocation) 
                    print("File:", filename + " - License Plate Number identified:", 
                                            licensePlateNumber)
                  

    correct_percentage = round((total/len(files))*100, 2)
    print("Total Images Processed:", len(files), "- Correctly Identified:", total, 
        "(" + str(correct_percentage) + "%)")
    

    
