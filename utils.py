import os
import time
import uuid
import numpy as np
import cv2

def stackImages(imgArray,scale,lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    newWidth = int(width * scale)
    newHeight = int(height * scale)
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (newWidth, newHeight))
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (newWidth, newHeight))
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                cv2.putText(ver,lables[d],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver

def rectContour(contours):
    rectCon = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if len(approx) == 4:
                rectCon.append(i)
    rectCon = sorted(rectCon, key=cv2.contourArea, reverse=True)
    return rectCon

def getCornerPoints(cont):
    peri = cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, 0.02 * peri, True)
    return approx

def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
   # print(myPoints)
    #print(add)
    #smallest one will be top left
    myPointsNew[0] = myPoints[np.argmin(add)]
    #largest one will be bottom right
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    #second smallest will be top right
    myPointsNew[1] = myPoints[np.argmin(diff)]
    #second largest will be bottom left
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew


# def splitBoxes(img):
#     rows = np.vsplit(img,10)
#     cv2.imshow('Split',rows[0])
#     boxes = []
#     for r in rows:
#         cols= np.hsplit(r,4)
#         cv2.imshow('Spl4it',cols[0])
#         for box in cols:
#             boxes.append(box)

#     return boxes


#get one  answer box from the image
def verticalSplitBoxes(img):
    #split image into 30 columns
    rows = np.hsplit(img,30)
    cv2.imshow('Split',rows[0])
    boxes = []
    for r in rows:
        #split each column into 10 answer rows
        cols= np.vsplit(r,20)
        #cv2.imshow('Spl4it',cols[9])
        for box in cols:
            #add each answer box to the list
           # cv2.imshow('Spl4it',box)
            boxes.append(box)
    return boxes

#divide answer boxes into 7 parts
def getAnswerBlocks(img):
    #reshape the image 
    img=cv2.resize(img,(64,64))
    #cv2.imshow('blocks',img)

   
    return img


#save images in new folder with improved classification
def saveImages(answerBlocks, boxNumber, predicted_digit=None, base_path=None):
    """
    Save image blocks to appropriate digit folders
    
    Args:
        answerBlocks: List of image blocks or single image block
        boxNumber: Sequential number for the box
        predicted_digit: If known, the digit class (0-9) to save to
        base_path: Base path for dataset, defaults to current dataset folder
    """
    if base_path is None:
        base_path = r"F:\University\fyp\digit_detection_model\dartaset\2"
    
    # Handle single image block
    if not isinstance(answerBlocks, list):
        answerBlocks = [answerBlocks]
    
    for i, block in enumerate(answerBlocks):
        if block is None or block.size == 0:
            continue
            
        # Analyze the image to determine if it's likely empty or contains a digit
        totalPixels = cv2.countNonZero(block)
        blockHeight, blockWidth = block.shape[:2]
        totalArea = blockHeight * blockWidth
        fillRatio = totalPixels / totalArea if totalArea > 0 else 0
        
        # Generate a unique filename
        unique_id = str(uuid.uuid4())[:8]
        timestamp = int(time.time())
        filename = f"box_{boxNumber:03d}_{i:02d}_{timestamp}_{unique_id}.jpg"
        
        # Save directly to base_path without creating subdirectories
        save_folder = base_path

        # Ensure the target directory exists
        os.makedirs(save_folder, exist_ok=True)
        
        # Save the image
        file_path = os.path.join(save_folder, filename)
        
        try:
            success = cv2.imwrite(file_path, block)
            if success:
                print(f"Saved: {file_path} (Fill ratio: {fillRatio:.3f})")
            else:
                print(f"Failed to save: {file_path}")
        except Exception as e:
            print(f"Error saving {filename}: {e}")


#save all 600 split images with proper organization
def saveAllSplitImages(boxes, base_path=None):
    """
    Save all 600 split images with proper organization
    
    Args:
        boxes: List of 600 image boxes from verticalSplitBoxes
        base_path: Base path for dataset
    """
    if base_path is None:
        base_path = r"F:\University\fyp\digit_detection_model\dartaset"
    
    print(f"Processing {len(boxes)} image boxes...")
    
    # Create the base directory
    os.makedirs(base_path, exist_ok=True)
    
    successful_saves = 0
    
    for box_idx, box in enumerate(boxes):
        try:
            # Process each box
            processed_block = getAnswerBlocks(box)
            
            # Save with proper indexing
            saveImages(processed_block, box_idx, base_path=base_path)
            successful_saves += 1
            
            # Progress indicator
            if (box_idx + 1) % 100 == 0:
                print(f"Processed {box_idx + 1}/{len(boxes)} boxes...")
                
        except Exception as e:
            print(f"Error processing box {box_idx}: {e}")
    
    print(f"Successfully saved {successful_saves}/{len(boxes)} images")
    return successful_saves


#manual classification helper for unclassified images
def manualClassifyImages(base_path=None):
    """
    Helper function to manually classify unclassified images
    Shows each image and asks for digit classification
    """
    if base_path is None:
        base_path = r"F:\University\fyp\digit_detection_model\dartaset"
    
    unclassified_path = os.path.join(base_path, "unclassified")
    
    if not os.path.exists(unclassified_path):
        print("No unclassified folder found")
        return
    
    files = [f for f in os.listdir(unclassified_path) if f.endswith('.jpg')]
    
    if not files:
        print("No unclassified images found")
        return
    
    print(f"Found {len(files)} unclassified images")
    print("Press keys 0-9 to classify, 'd' to delete, 's' to skip, 'q' to quit")
    
    for filename in files:
        file_path = os.path.join(unclassified_path, filename)
        img = cv2.imread(file_path)
        
        if img is None:
            continue
            
        # Resize for better viewing
        display_img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_NEAREST)
        cv2.imshow(f"Classify: {filename}", display_img)
        
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        
        if key == ord('q'):  # Quit
            break
        elif key == ord('s'):  # Skip
            continue
        elif key == ord('d'):  # Delete
            os.remove(file_path)
            print(f"Deleted {filename}")
        elif ord('0') <= key <= ord('9'):  # Classify as digit
            digit = chr(key)
            target_folder = os.path.join(base_path, digit)
            os.makedirs(target_folder, exist_ok=True)
            
            target_path = os.path.join(target_folder, filename)
            os.rename(file_path, target_path)
            print(f"Moved {filename} to digit {digit} folder")
        else:
            print("Invalid key pressed, skipping...")


#analyze dataset distribution
def analyzeDataset(base_path=None):
    """
    Analyze the distribution of images across digit folders
    """
    if base_path is None:
        base_path = r"F:\University\fyp\digit_detection_model\dartaset\2"
    
    print("Dataset Analysis:")
    print("-" * 40)
    
    total_images = 0
    
    for digit in range(10):
        digit_path = os.path.join(base_path, str(digit))
        if os.path.exists(digit_path):
            count = len([f for f in os.listdir(digit_path) if f.endswith('.jpg')])
            print(f"Digit {digit}: {count} images")
            total_images += count
        else:
            print(f"Digit {digit}: 0 images (folder doesn't exist)")
    
    # Check special folders
    special_folders = ["unclassified", "unclear"]
    for folder in special_folders:
        if os.path.exists(base_path):
            count = len([f for f in os.listdir(base_path) if f.endswith('.jpg')])
            print(f"{folder.capitalize()}: {count} images")
            total_images += count
    
    print("-" * 40)
    print(f"Total images: {total_images}")
    
    return total_images


#show answers on the image
def showAnswers(img,answerIndexes):
    secW = int(img.shape[1] / 35)
    secH = int(img.shape[0] / 10)
    print(secH,secW,img.shape)

    for x in range(0,5):

        for y in range(0,10):
            myAns = answerIndexes[x*10+y]
            if myAns != -1:
                cx=(myAns*secW +7*x*secW)+secW//2
                cy=(y*secH)+secH//2
                cv2.rectangle(img,(cx-10,cy-10),(cx+10,cy+10),(0,255,0),cv2.FILLED)

    return img

# def extract_text_from_box(img, contour):
#     contour = reorder(contour)
#     pt1 = np.float32(contour)
#     pt2 = np.float32([[0, 0], [350, 0], [0, 100], [350, 100]])
#     matrix = cv2.getPerspectiveTransform(pt1, pt2)
#     imgWarp = cv2.warpPerspective(img, matrix, (350, 100))

#     imgGray = cv2.cvtColor(imgWarp, cv2.COLOR_BGR2GRAY)
#     _, imgThresh = cv2.threshold(imgGray, 150, 255, cv2.THRESH_BINARY)

#     config = '--psm 6'  # Assume a single uniform block of text
#     text = pytesseract.image_to_string(imgThresh, config=config)
#     return text.strip(), imgWarp