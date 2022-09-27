from turtle import left
import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils
import pytesseract
import re

from os import listdir


DIR_IMAGES = 'images/'
DEBUG = True


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

            print(f'\nRatio: {ratio} ({y}/{x})\n')

            if 4 > ratio > 1.8:
                location = approx
                break

    return location


def preprocess(image, edge_args):
    
    alpha, beta = 1.5, 0
    contrast_img = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    edged = cv2.Canny(image, edge_args[0], edge_args[1])
    cv2.imshow('pepe', edged)
    cv2.waitKey(0)
    location = get_location(edged.copy())
    
    if location is None:
        print('Location not found!')
        return 

    mask = np.zeros(image.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0,255, -1)
    new_image = cv2.bitwise_and(image, image, mask=mask)

    cv2.imshow(new_image)

    (x, y) = np.where(mask==255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = new_image[x1:x2+1, y1:y2+1]

    cv2.imshow('final image', cropped_image)
    return read_number_plate(cropped_image)


def metode_A(image):

    alpha, beta = 1.5, 0
    contrast_img = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    blurred = cv2.GaussianBlur(contrast_img, (3,3), 0)
    cv2.imshow('blurred', blurred)
    cv2.waitKey(0)

    bin_img = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV, 11, 10)
    cv2.imshow('binary', bin_img)
    cv2.waitKey(0)

    edged = cv2.Canny(bin_img, 30, 200)
    cv2.imshow('edged', edged)
    cv2.waitKey(0)

    img = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, (3,3), iterations=5)
    cv2.imshow('edged + morph', img)
    cv2.waitKey(0)

    location = get_location(img.copy())
    
    if location is None:
        print('Location not found!')
        return 

    mask = np.zeros(image.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0,255, -1)
    new_image = cv2.bitwise_and(image, image, mask=mask)

    cv2.imshow('pepe', new_image)
    cv2.waitKey(0)

    (x, y) = np.where(mask==255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = new_image[x1:x2+1, y1:y2+1]

    cv2.imshow('final image', cropped_image)
    cv2.waitKey(0)

    return cropped_image
    


def metode_B(image):
    img = cv2.GaussianBlur(image, (7,7), 0)
    cv2.imshow('Blur', img)
    cv2.waitKey(0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    img = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel, iterations=2)
    cv2.imshow('Black hat', img)
    cv2.waitKey(0)

    img = cv2.threshold(img, img.max()//2, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow('THS', img)
    cv2.waitKey(0)
  
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,1))
    img = cv2.dilate(img, horizontal_kernel, iterations=5)
    cv2.imshow('dilate', img)
    cv2.waitKey(0)

    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
    cv2.imshow('Open', img)
    cv2.waitKey(0)

    img = cv2.GaussianBlur(img, (3,3), 0)
    img = cv2.threshold(img, img.max()//2, 255, cv2.THRESH_BINARY)[1]

    square_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, square_kernel, iterations=5)
    cv2.imshow('Close', img)
    cv2.waitKey(0)

    # Filter using contour area and remove small noise
    cnts = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 1000:
            cv2.drawContours(img, [c], -1, (0,0,0), -1)

    cv2.imshow('cnt', img)
    cv2.waitKey(0)

    location = get_location(img.copy())
    
    if location is None:
        print('Location not found!')
        return 

    mask = np.zeros(image.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0,255, -1)
    new_image = cv2.bitwise_and(image, image, mask=mask)

    cv2.imshow('pepe', new_image)
    cv2.waitKey(0)
    
    warped = homography(new_image,location)
    cv2.imshow("warped",warped)
    cv2.waitKey(0)
    return warped



def read_number_plate(image):
    # Read the number plate
    text = pytesseract.image_to_string(image, config='--psm 11')
    text = re.sub('[^0-9a-zA-Z]+', '*', text)
    print("Detected license plate Number is:", text)
    print(repr(text))
    return text


def elimina_e(img, thr):
  mean_columns = img.mean(0)

  for ind, val in enumerate(mean_columns):
    if val > thr:
      break
    
  img_ret = np.delete(img, [i for i in range(ind)], axis=1)

  return img_ret

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

    files = [DIR_IMAGES + file for file in listdir(DIR_IMAGES)]

    num_img = np.random.randint(0, len(files))
    #num_img = 79
    img = read_image(files[num_img])

    y, x = img.shape

    print(f'Imatge num: {num_img}')

    # Resize Image
    r = 720 / x
    dim = (720, int(y * r))
    img = cv2.resize(img, dim)

    y, x = img.shape
    img_cropped = img[y//4:y, 0:x]

    if DEBUG:
        cv2.imshow(files[num_img], img)
        cv2.imshow('img_cropped', img_cropped)
        cv2.waitKey(0)
    
    metode_B(img_cropped)

    real_NP = files[num_img]
    #print(f'Real NP: {real_NP} \nNP founded: {number_plate}')
