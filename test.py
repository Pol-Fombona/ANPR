from distutils.log import debug
from msilib.schema import Directory
from turtle import left
import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils
import pytesseract
import re
import os

from os import listdir



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
        #approx = cv2.approxPolyDP(contour, 35, True)
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

            if 4 > ratio > 1.8:
                location = approx
                break

    return location

def show_image(img, title = ''):

    #img = ResizeWithAspectRatio(img, 720)
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def metode_B(image):

    history = []

    img = cv2.GaussianBlur(image, (7,7), 0) 
    # Compte que aqui li apliquem el blur pero despres al black hat usem la imatge de
    # parametre i no la del gaussian
    history.append((img, "Gaussian Blur"))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    blackHat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel, iterations=2)
    history.append((blackHat, "Black Hat"))

    treshold = cv2.threshold(blackHat, blackHat.max()//2, 255, cv2.THRESH_BINARY)[1]
    history.append((treshold, "Treshold"))
  
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,1))
    dilate = cv2.dilate(treshold, horizontal_kernel, iterations=5)
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
    cnts = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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

    mask = np.zeros(image.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0,255, -1)
    new_image = cv2.bitwise_and(image, image, mask=mask)


    (x, y) = np.where(mask==255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = new_image[x1:x2+1, y1:y2+1]

    warped = homography(new_image,location)
    history.append((warped, "Warped"))

    if (DEBUG):
        for item in history:
            show_image(item[0], item[1])

    return warped

def img_to_str(img):
    #show_image(img)
    custom_config = r'-c tessedit_char_whitelist=0123456789BCDFGHJKLMNPRSTVWXYZ --psm 11'
    ## l'11 es el que funciona millor pero hauriem d'usar el 7
    ## El 10 si tenim els caracters separats

    text = pytesseract.image_to_string(img, config=custom_config)

    text = text.replace("\n", "").replace(" ", "")

    imgBoxes = getTextBoxes(img)

    return text, imgBoxes

def getTextBoxes(img):

    boxes = pytesseract.image_to_boxes(img)

    height, width = img.shape

    for box in boxes.splitlines():
        box = box.split(' ')

        x, y, w, h = int(box[1]), int(box[2]), int(box[3]), int(box[4])
        cv2.rectangle(img, (x, height - y), (w, height - h), (50, 50, 255), 1)
        cv2.putText(img, box[0], (x, height - y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 205, 50), 1)

    return img



def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")

	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect

def homography(gray_img,location):
	src = order_points(np.squeeze(location).astype(np.float32))
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


if __name__ == '__main__':

    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    directory = os.getcwd()
    
    DIR_IMAGES = directory + "\cropped_images\\"
    DIR_RESULTS = directory + "\licensePlate\\"

    DEBUG = False
    total = 0

    files = [DIR_IMAGES + file for file in listdir(DIR_IMAGES)]
    files = files[:20]
    
    for file in files:

        img = read_image(file)

        y, x = img.shape

        # Resize Image
        r = 720 / x
        dim = (720, int(y * r))
        img = cv2.resize(img, dim)

        y, x = img.shape
        #img_cropped = img[y//4:y, 0:x]
        img_cropped = img
        plateLocation = metode_B(img_cropped)
        file_name = file[file.rindex('\\') + 1 :]
        filePlateTag = file_name[:file_name.rindex('.')]

        if (plateLocation is None):
            print("File:", file_name, "- License Plate not found")          

        else:
            

            licensePlateNumber, imgTextBoxes = img_to_str(plateLocation)

            cv2.imwrite(DIR_RESULTS + file_name, imgTextBoxes)

            
            print("File:", file_name + " - License Plate Number identified: ", 
                licensePlateNumber)

            if licensePlateNumber == filePlateTag: 
                total += 1

    correct_percentage = round((total/len(files))*100, 2)
    print("Total Images Processed:", len(files), "- Correctly Identified:", total, 
        "(" + str(correct_percentage) + "%)")

    
