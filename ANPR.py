from imutils import contours
from os import listdir

import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils
import re
import os
import sys
import easyocr


def read_image(filename):
    # Read Image in Gray Scale & Color
    return cv2.imread(filename, 0),cv2.imread(filename)

def getLocation(image):
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
    # Shows image in a window
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def localization(image, og_img, img_color):

    history = []

    imgTransformed = transformations(image, history)
    imgContours = getContours(history, imgTransformed)
    location = getLocation(imgContours.copy())
    
    if location is None:
        if (DEBUG): print('Location not found!')
        return 

    itemp = img_color.copy()
    loc = location.copy()

    x = location[0][0]
    y = location[0][1]
    w = location[2][0] - x
    h = location[3][1] - y

    cv2.drawContours(itemp, [location], -1, (124,252,0), 2)

    mask = np.zeros(og_image.shape, np.uint8)
    y, x = og_image.shape

    for i in range(4):
        location[i][0] = location[i][0] / (720/x)
        location[i][1] = location[i][1] / (720/x)

    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(og_image, og_image, mask=mask)

    warped = homography(new_image, location)
    history.append((warped, "Warped"))

    _, imgTreshold = cv2.threshold(warped, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    imgLicensePlate = removeEUStrip(imgTreshold, 120)

    history.append((imgLicensePlate, "Removed EU strip"))

    if (DEBUG):
        for item in history:
            show_image(item[0], item[1])

    return imgLicensePlate, itemp, loc

def getContours(history, imgTransformed):
    # Filter using contour area and remove small noise

    cnts = cv2.findContours(imgTransformed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 1000:
            cv2.drawContours(imgTransformed, [c], -1, (0,0,0), -1)

    history.append((imgTransformed, "Contours"))

    return imgTransformed


def transformations(image, history):

    blurred = cv2.GaussianBlur(image, (5,5), 0) 
    history.append((blurred, "Gaussian Blur"))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    blackHat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel, iterations=3)
    history.append((blackHat, "Black Hat"))

    treshold = cv2.threshold(blackHat, blackHat.max()//2, 255, cv2.THRESH_BINARY)[1]
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
    
    return closing


def imgToText(img, reader):
    # Returns text detected in the image

    result = reader.readtext(img, detail=0, allowlist = '0123456789BCDFGHJKLMNPRSTVWXYZ', paragraph=True)

    if len(result) == 0:
        result = ''
    
    elif len(result) >= 1:
        result = ''.join(result).replace(" ", "")
    
    return result


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
    # Corrects rotation
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
    # Removes the blue strip
    mean_columns = img.mean(0)

    for ind, val in enumerate(mean_columns):
        if val > thr:
            break

    img_ret = np.delete(img, [i for i in range(ind)], axis=1)

    return img_ret


def checkText(text):
    # Post process results of OCR

    if len(text) > 7:
        # If len of text is bigger than expected split parts in numbers and text
        split_text = (re.split('(\d+)',text))

        numbers = split_text[1]
        letters = split_text[2]

        # IF more than 4 numbers remove the the first one and keep the rest
        if len(numbers) > 4:
            numbers = numbers[1:]

        # If more than 3 text characters, keep only the first three
        if len(letters) > 3:
            letters = letters[:3]

        text = numbers + letters

    # Replace similar characters and numbers
    number_text_similarity = (("L","4"), ("S", "5"), ("Z","2"), ("B","8"), ("G","6"))
    numbers, letters = text[:4], text[4:]

    for pair in number_text_similarity:
        numbers = numbers.replace(pair[0], pair[1])
        letters = letters.replace(pair[1], pair[0])

    text = numbers + letters

    return text


def drawBoundingBoxResult(img_boundingBox, location, licensePlateNumber):

    img_boundingBox = cv2.putText(img_boundingBox, licensePlateNumber, (location[0][0]+50, location[0][1] - 50),
			cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 10)
    img_boundingBox = cv2.putText(img_boundingBox, licensePlateNumber, (location[0][0]+50, location[0][1] - 50),
			cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    return img_boundingBox


def saveResult(directory, filename, plateLocation, licensePlateNumber, correct):
    
    cv2.imwrite(directory + filename, plateLocation)
    result = "File: " + filename + " - License Plate Number identified: " + licensePlateNumber

    result += " ✓" if correct else " ✕"

    print(result)


def resizeImg(img):

    shape = img.shape
    y, x = shape[0], shape[1]

    r = 720 / x
    dim = (720, int(y * r))
    img = cv2.resize(img, dim)

    return img


if __name__ == '__main__':

    directory = os.getcwd()
    
    DIR_IMAGES = directory + "\\images\\"
    
    DIR_CORRECT_RESULTS = directory + "\licensePlate\correct\\"
    DIR_INCORRECT_RESULTS = directory + "\licensePlate\incorrect\\"
    DIR_IMAGES_BB = directory + "\licensePlate\\boundingBox\\"

    directories = [DIR_CORRECT_RESULTS, DIR_INCORRECT_RESULTS, DIR_IMAGES_BB]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

    DEBUG = False
    if len(sys.argv) >= 2:
        if sys.argv[1] == "--debug":
            DEBUG = True
            print("### Using debug mode, each image transformation will be shown. ###")

    reader = easyocr.Reader(['en'], gpu = False)
    files = listdir(DIR_IMAGES)
    total = 0

    for filename in files:

        img,img_color = read_image(DIR_IMAGES + filename)   
        og_image = img.copy()
        
        img = resizeImg(img)
        img_color = resizeImg(img_color)

        try:

            plateLocation, img_boundingBox, location = localization(img, og_image,img_color)

            if "_" in filename:
                filePlateTag = filename[:filename.rindex('_')]
            else:
                filePlateTag = filename[:filename.rindex('.')]

            licensePlateNumber = imgToText(plateLocation, reader)
            
            if licensePlateNumber == filePlateTag:
                # Correctly recognized characters
                saveResult(DIR_CORRECT_RESULTS, filename, plateLocation, licensePlateNumber, True)
                total += 1

            else:
                licensePlateNumber = checkText(licensePlateNumber)

                if licensePlateNumber == filePlateTag:
                    # Correctly recognized characters after post-process of the first result
                    saveResult(DIR_CORRECT_RESULTS, filename, plateLocation, licensePlateNumber, True)
                    total += 1

                else:
                    # Incorrectly recognized characters
                    saveResult(DIR_INCORRECT_RESULTS, filename, plateLocation, licensePlateNumber, False)
            
            # Save image with the bounding box and text recognized
            img_boundingBox = drawBoundingBoxResult(img_boundingBox, location, licensePlateNumber)
            cv2.imwrite(DIR_IMAGES_BB + filename, img_boundingBox)
        
        except:
            print("File:", filename, "- License Plate not found") 
            continue  

    correct_percentage = round((total/len(files))*100, 2)
    print("Total Images Processed:", len(files), "- Correctly Identified:", total, 
        "(" + str(correct_percentage) + "%)")
    

    
