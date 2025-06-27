import cv2
import numpy as np

IMAGE_PATHS = ["images/" + path for path in [
    "Example_Image_1.jpg", 
    "Example_Image_2-1024x681.jpg", 
    "Example_Image_3-1024x512.png", 
    "Example_Image_4.png", 
    "Example_Receipt_Image.jpeg"]
]
MODEL_PATHS = {key: "models/" + value for key, value in ({
    "EAST": "frozen_east_text_detection.pb",
    "DB50": "DB_TD500_resnet50.onnx",
    "DB18": "DB_TD500_resnet18.onnx"
}).items()}
DEFAULT_SCALE = 1.0
DEFAULT_MEAN = (122.67891434, 116.66876762, 104.00698793)
INPUT_SIZE = (320, 320)

def getImageFromFilePath(imagePath: str, inputSize: tuple[int, int]):
    return cv2.resize(cv2.imread(imagePath), inputSize)

def configureEASTTextDetector(textDetector, confThresh: float, nmsThresh: float, inputSize: tuple[int, int], mean: tuple[float, float, float]) -> None:
    textDetector.setConfidenceThreshold(confThresh)
    textDetector.setNMSThreshold(nmsThresh)
    textDetector.setInputParams(DEFAULT_SCALE, inputSize, mean, True)

def configureDBTextDetector(textDetector, binThresh: float, polyThresh: float, inputSize: tuple[int, int], mean: tuple[float, float, float]) -> None:
    textDetector.setBinaryThreshold(binThresh)
    textDetector.setPolygonThreshold(polyThresh)
    textDetector.setInputParams(DEFAULT_SCALE/255, inputSize, mean, True)

def detectText(origImage, textDetector):
    boxes, confidences = textDetector.detect(origImage)
    return boxes

def main():
    origImage = getImageFromFilePath(IMAGE_PATHS[-1], INPUT_SIZE)
    runs = [
        {
            "name": "EAST",
            "textDetector": cv2.dnn_TextDetectionModel_EAST(MODEL_PATHS["EAST"]), # type: ignore
            "configureTextDetector": lambda textDetector: configureEASTTextDetector(textDetector, 0.8, 0.4, INPUT_SIZE, DEFAULT_MEAN),
            "colorTuple": (0, 255, 255) # cyan
        }, 
        {
            "name": "DB50",
            "textDetector": cv2.dnn_TextDetectionModel_DB(MODEL_PATHS["DB50"]), # type: ignore
            "configureTextDetector": lambda textDetector: configureDBTextDetector(textDetector, 0.3, 0.5, INPUT_SIZE, DEFAULT_MEAN),
            "colorTuple": (0, 0, 255) # red
        },
        {
            "name": "DB18",
            "textDetector": cv2.dnn_TextDetectionModel_DB(MODEL_PATHS["DB18"]), # type: ignore
            "configureTextDetector": lambda textDetector: configureDBTextDetector(textDetector, 0.3, 0.5, INPUT_SIZE, DEFAULT_MEAN),
            "colorTuple": (255, 0, 0) # blue
        }
    ]
    
    annotatedImages = [origImage.copy() for _ in runs]
    for index, run in enumerate(runs):
        run["configureTextDetector"](run["textDetector"])
        boxes = detectText(origImage, run["textDetector"])
        for box in boxes:
            cv2.polylines(annotatedImages[index], [np.array(box, np.int32)], isClosed=True, color=run["colorTuple"], thickness=1)

    cv2.imshow('Original', origImage)
    for index, run in enumerate(runs):
        cv2.imshow(f'Annotated {run["name"]}', annotatedImages[index])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()