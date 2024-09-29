# inital import of libraries 
import cv2
import numpy as np


# using cv2 , numpy for plotting and math 

'''
Initial Intuition: To use pre-trained data of cones and image detection library to detect the cones
only need TWO cones from each side to be able to be able to create a linear line such as the one in the image 
Was going to use SSD MobileNet V2 FPNLite 320x320 (22MB) from the tensorFlow Detection model zoo
This model  because its a lot less storage than other dataset models and should still work fine for this task
Then realize it does not contain the label of 'cones' or 'orange cones' which means i would have to train it 
myself with thousands of images 
'''
# new method: using color detection since orange is a more rare color, detecting orange clusters could work 

image = cv2.imread('original.png')

# making the picture HSV (hue, saturation, value) for detection - internet says this is better for detecting oranges 
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
'''
Define Color range for orange
getting colors from: https://colorizer.org/
started with a wide range of oranges to idenity our boxes and trimmed so it only inludes the bright orange cones
not the other orange looking things in the picture, ending up being pretty small which works for this (prob not super dynamic)
'''
lower_orange = np.array([0, 100, 100])
upper_orange = np.array([3, 255, 255])

# creating a mask to isolate orange colors
mask = cv2.inRange(hsv_image, lower_orange, upper_orange)

# getting our contours back, only need that value
contours, irrelevent = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)




def getCenters(contours):
    # setting a min area for the boxes so it disregards the little spec boxes 
    # i found that the smallest 2 cones were around 7 x 10 so i did min area of 50 (little smaller)
    min_area = 50
    # get centers for lefter and right blocks 
    left_centers = [] 
    right_centers = []

    for contour in contours:
        # get area 
        box_area = cv2.contourArea(contour) 

        # if the area is big enough, store the centers 
        if box_area > min_area:      
            # Get the bounding rectangle and boxes 
            coordinates = cv2.boundingRect(contour) 

            x = coordinates[0]
            y = coordinates[1]
            w = coordinates[2]
            h = coordinates[3]
            
            # Calculate center coordinates
            center_x = x + w // 2
            center_y = y + h // 2
            
           # add the centers to the list
           # Store in left or right centers based on x-coordinate
            if center_x < image.shape[1] // 2:  # Left side
                left_centers.append([center_x, center_y])
            else:  # Right side
                right_centers.append([center_x, center_y])

    return left_centers, right_centers



def getLineEndpoints(left_centers, right_centers):
    def calculate_endpoints(centers):
        center_count = len(centers)
        #grab our bottom and top cones 
        x1, y1 = centers[center_count - 1]  
        x2, y2 = centers[0]  

        # Calculate slope
        slope = (y2 - y1) / (x2 - x1)

        height = image.shape[0]
        width = image.shape[1]

        # Calculate x at y = 0 (top of image)
        x_start = int(x1 - (y1 / slope)) 
        y_start = 0 # alwats 

        # Calculate x at y = height (bottom of image)
        x_end = int(x1 + ((height - y1) / slope))

        # Calculate y at x = 0 (left edge)
        y_at_left = int(slope * (0 - x1) + y1)
        # Calculate y at x = width (right edge)
        y_at_right = int(slope * (width - x1) + y1)

        # Handle different edge cases like line through the top, bottom, left, or right
        # line can either come in through top or the sides, and then exit at bottom or sides
        if 0 <= x_start <= width:  # Line crosses through the top
            start_point = (x_start, y_start)
            if x_end >= 0 and x_end <= width:  # Line also crosses through the bottom
                end_point = (x_end, height - 1)
            elif y_at_right >= 0 and y_at_right <= height:  # Line crosses through the right side
                end_point = (width - 1, y_at_right)
            elif y_at_left >= 0 and y_at_left <= height:  # Line crosses through the left side
                end_point = (0, y_at_left)
        elif y_at_left >= 0 and y_at_left <= height:  # Line crosses through the left side
            start_point = (0, y_at_left)
            if y_at_right >= 0 and y_at_right <= height:  # Line crosses through the right side
                end_point = (width - 1, y_at_right)
            elif x_end >= 0 and x_end <= width:  # Line crosses through the bottom
                end_point = (x_end, height - 1)
        else:
            print("Something went wrong")

        return start_point, end_point

    # get out points 
    left_endpoints = calculate_endpoints(left_centers)
    right_endpoints = calculate_endpoints(right_centers)

    return left_endpoints, right_endpoints


# draw our lines 
def drawLines(left_centers, right_centers):
    left_endpoints, right_endpoints = getLineEndpoints(left_centers, right_centers)

    #  left line
    cv2.line(image, left_endpoints[0], left_endpoints[1], (0, 0, 255), 2)  # red = 0,0,255
    # right line
    cv2.line(image, right_endpoints[0], right_endpoints[1], (0, 0, 255), 2) 


# actually displaying lines 
left_centers, right_centers = getCenters(contours)
drawLines(left_centers, right_centers)

cv2.imshow("Detected Cones", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


