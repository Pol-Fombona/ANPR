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
    keypoints = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    num_contours = 10 if len(contours) > 10 else len(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:num_contours]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 35, True)

        if len(approx) == 4:
            
            points = np.squeeze(approx)
            
            X, Y = points[:,0], points[:,1]

            left_y, right_y = np.min(Y), np.max(Y)
            top_x, bottom_x = np.min(X), np.max(X)
            
            x, y = right_y - left_y, bottom_x - top_x
            ratio = y / x
            print(f'\nRatio: {ratio} ({y}/{x})\n')

            color = (255, 0, 0)
            thickness = 4
            #nimage = cv2.rectangle(image, points[0], points[-1], color, thickness)
            pts = points.reshape((-1, 1, 2))
            nimage = cv2.polylines(image, [pts], True, color, thickness)      

            # Displaying the image 
            cv2.imshow('pepe', nimage)
            cv2.waitKey(0)

            if ratio > 1.5:
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
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

    img = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel, iterations=2)
    cv2.imshow('Black hat', img)
    cv2.waitKey(0)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=7)
    cv2.imshow('closing', img)
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


def read_number_plate(image):
    # Read the number plate
    text = pytesseract.image_to_string(image, config='--psm 11')
    text = re.sub('[^0-9a-zA-Z]+', '*', text)
    print("Detected license plate Number is:", text)
    print(repr(text))
    return text


if __name__ == '__main__':

    files = [DIR_IMAGES + file for file in listdir(DIR_IMAGES)]

    #num_img = np.random.randint(0, len(files))
    num_img = 20
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

    #number_plate = preprocess(img_cropped, (30, 200))
    
    metode_B(img_cropped)
    real_NP = files[num_img]

    #print(f'Real NP: {real_NP} \nNP founded: {number_plate}')
