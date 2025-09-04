import utils
import cv2
import numpy as np
import test_model as model

try:
    # Load image
    image = cv2.imread(r"F:\University\fyp\digit_detection_model\images\test.jpg")
    cv2.imshow("Original", image)

    # Resize and preprocess
    img = cv2.resize(image, (640, 64))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 1)
    edges = cv2.Canny(blur, 10, 50)
    cv2.imshow("Edges", edges)

    # Contour detection
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rects = utils.rectContour(contours)
    if not rects:
        raise ValueError("No contours found")

    # Perspective transform
    biggest = utils.getCornerPoints(rects[0])
    biggest = utils.reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [640, 0], [0, 64], [640, 64]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(img, matrix, (640, 64))

    # Thresholding
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        warped_gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # Morphological operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    erosion_image = cv2.erode(thresh, kernel, iterations=1)
    cv2.imshow("Eroded", erosion_image)

    # Split into 10 boxes
    boxes = np.hsplit(erosion_image, 10)
    
    print(f"Number of boxes: {len(boxes)}")  # Should print 10

    # Prepare boxes for model - convert grayscale to RGB and resize properly
    model_inputs = []
    for i, box in enumerate(boxes):
        # Resize to 64x64
        resized_box = cv2.resize(box, (64, 64))
        
        # Convert grayscale to RGB (3 channels)
        if len(resized_box.shape) == 2:  # Grayscale
            rgb_box = cv2.cvtColor(resized_box, cv2.COLOR_GRAY2RGB)
        else:
            rgb_box = resized_box
            
        model_inputs.append(rgb_box)
        
        # Save for debugging
        cv2.imwrite(f"F:/University/fyp/digit_detection_model/images/test/box_{i}.png", resized_box)
    # Classify batch
    predictions = model.classify_batch(model_inputs)  # returns list of (label, confidence)

    # Print predictions
    for i, (label, confidence) in enumerate(predictions):
        print(f"Box {i}: Predicted {label}, Confidence: {confidence}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

except Exception as e:
    print("Error during processing:", e)
