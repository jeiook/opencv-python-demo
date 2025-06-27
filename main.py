import cv2
import numpy as np

IMAGE_PATHS = ["Example_Image_1.jpg", "Example_Image_2-1024x681.jpg", "Example_Image_3-1024x512.png", "Example_Image_4.png"]
MODEL_PATH = "frozen_east_text_detection.pb"
DEFAULT_SCALE = 1.0
DEFAULT_MEAN = (122.67891434, 116.66876762, 104.00698793)
INPUT_SIZE = (320, 320)

def imagePathToImage(imagePath: str, inputSize: tuple[int, int]):
    return cv2.resize(cv2.imread(imagePath), inputSize)

def detectText(origImage, confThresh: float, nmsThresh: float, inputSize: tuple[int, int], mean: tuple[float, float, float]):
    textDetectorEAST= cv2.dnn_TextDetectionModel_EAST(MODEL_PATH) # type: ignore
    textDetectorEAST.setConfidenceThreshold(confThresh)
    textDetectorEAST.setNMSThreshold(nmsThresh)
    textDetectorEAST.setInputParams(DEFAULT_SCALE, inputSize, mean, True)
    boxes, confidences = textDetectorEAST.detect(origImage)
    return boxes

def main():
    origImage = imagePathToImage(IMAGE_PATHS[3], INPUT_SIZE)
    annotated_image = origImage.copy()
    boxes = detectText(origImage, 0.8, 0.4, INPUT_SIZE, DEFAULT_MEAN)
    for box in boxes:
        cv2.polylines(annotated_image, [np.array(box, np.int32)], isClosed=True, color=(0, 255, 255), thickness=1)  # Annotate EAST (Cyan)

    cv2.imshow('Original', origImage)
    cv2.imshow('Annotated', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()