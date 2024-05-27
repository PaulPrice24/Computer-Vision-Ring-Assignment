import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import convolve2d
from collections import deque

# Implement thresholding ourselves using loops (soooo slow in Python)
def threshold(img, thresh):
    for y in range(0, img.shape[0]):  # Loop through rows
        for x in range(0, img.shape[1]):  # Loop through columns
            if img[y, x] < thresh:
                img[y, x] = 255
            else:
                img[y, x] = 0

#find histogram of image
def imhist(img):
    hist = np.zeros(256)
    for y in range(0, img.shape[0]):  # Loop through rows
        for x in range(0, img.shape[1]):  # Loop through columns
            hist[img[y, x]] += 1
    return hist

#find peak in histogram
def find_peak(hist):
    peak_count = 0
    peak_index = 0
    for i in range(hist.size):
        if hist[i] > peak_count:
            peak_count = hist[i]
            peak_index = i
    return peak_index

#perform morphological close
def morphological_close(img, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    #perform dilation and erosion
    dilated_img = custom_dilate(img, kernel)
    closed_img = custom_erode(dilated_img, kernel)
    return closed_img

#dilation function
def custom_dilate(img, kernel):
    dilated_img = np.zeros_like(img)
    pad_height = kernel.shape[0] // 2
    pad_width = kernel.shape[1] // 2
    padded_img = np.pad(img, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    #loop through each pixel in the image
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            #get region of interest (roi)
            roi = padded_img[y:y+kernel.shape[0], x:x+kernel.shape[1]]
            dilated_img[y, x] = np.max(roi * kernel)
    return dilated_img

#erosion funtion
def custom_erode(img, kernel):
    eroded_img = np.zeros_like(img)
    pad_height = kernel.shape[0] // 2
    pad_width = kernel.shape[1] // 2
    padded_img = np.pad(img, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    #loop through each pixel
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            #get roi
            roi = padded_img[y:y+kernel.shape[0], x:x+kernel.shape[1]]
            #apply erosion
            eroded_img[y, x] = np.min(roi * kernel)
    return eroded_img

#label connected components function
def connected_components(img):
    labeled_img = np.zeros_like(img)
    label = 1

    visited = np.zeros_like(img, dtype=bool)

    #loop through each pixel
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            #if pixel is foreground
            if img[y, x] == 255 and labeled_img[y, x] == 0 and not visited[y, x]:
                #assign colour to label
                color = colors[label]
                #perform flood fill
                flood_fill(img, labeled_img, x, y, label, color, visited)
                label += 1

    return labeled_img

#flood fill function
def flood_fill(img, labeled_img, x, y, label, color, visited):
    stack = deque([(x, y)])

    while stack:
        x, y = stack.pop()
        if x < 0 or y < 0 or x >= img.shape[1] or y >= img.shape[0] or visited[y, x] or img[y, x] == 0 or labeled_img[y, x] > 0:
            continue

        labeled_img[y, x] = label
        visited[y, x] = True

        stack.append((x + 1, y))
        stack.append((x - 1, y))
        stack.append((x, y + 1))
        stack.append((x, y - 1))

max_label = 100 
colors = np.random.randint(0, 255, size=(max_label + 1, 3)) 

#function to determine whether ring is broken
def analyze_regions(labeled_img):
    num_components = np.max(labeled_img)

    min_components_expected = 1  
    min_area_threshold = 1000   

    if num_components <= min_components_expected:
        print("Fail: Ring is broken")
    else:
        for label in range(1, num_components + 1):
            component_area = np.sum(labeled_img == label)
            if component_area < min_area_threshold:
                print("Fail: Ring is damaged")
                return
        
        print("Pass")

#main loop
for i in range(1, 16):
    start_time = time.time()
    img = cv.imread(f'C:\\Users\\paulj\\OneDrive\\Desktop\\Computer Vision Ring Assignment\\Orings\\Oring{i}.jpg', 0)
    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    thresh = np.argmax(hist) - 70
    print(thresh)
    cv.imshow('original image', img)
    end_time = time.time()
    processing_time = end_time - start_time
    
    _, img = cv.threshold(img, thresh, 255, cv.THRESH_BINARY)
    labeled_img = connected_components(img)
    analyze_regions(labeled_img)

    threshold(img,thresh)
    img2 = morphological_close(img, kernel_size=5)
    img_bgr = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.imshow('binary morphology image',img2)

    colored_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for label in range(1, np.max(labeled_img) + 1):
        colored_img[labeled_img == label] = colors[label]

    cv.imshow('labeled image', colored_img)
    _, img = cv.threshold(img, thresh, 255, cv.THRESH_BINARY)
    threshold(img,thresh)
    cv.putText(img, f"Time: {processing_time:.2f} seconds", (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)
    cv.putText(img, f"Threshold: {thresh}", (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)
    cv.imshow('thresholded image',img)
    cv.waitKey()
    plt.plot(hist)
    plt.show()

cv.destroyAllWindows()