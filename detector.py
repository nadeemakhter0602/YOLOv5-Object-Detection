import cv2
import onnxruntime as rt
import os
import numpy as np
import time

work_dir = os.path.dirname(os.path.realpath(__file__))

class Detector():

    def __init__(self, model_name, rescale, input_shape, backend='cpu', config=None):
        self.rescale = rescale
        self.input_shape = input_shape
        if backend=='cpu' or backend=='cuda':
            provider = ['CUDAExecutionProvider' if backend=='cuda' else 'CPUExecutionProvider']
            self.model = rt.InferenceSession(os.path.join(work_dir, model_name), providers=provider)
            self.input_name = self.model.get_inputs()[0].name
        elif backend=='cv2':
            self.model = cv2.dnn_DetectionModel(os.path.join(work_dir, model_name), config=config)
            self.model.setInputScale(scale=self.rescale)
            self.model.setInputSize(size=self.input_shape)

    def format_yolov5(self, frame):
        row, col, _ = frame.shape
        _max = max(col, row)
        result = np.zeros((_max, _max, 3), np.uint8)
        result[0:row, 0:col] = frame
        return result

    def detect(self, image):
        image = self.format_yolov5(image)
        
        if isinstance(self.model, cv2.dnn_DetectionModel):
            outputs = self.model.predict(image)
            # last model layer is of shape (1, 25500, 85)
            outputs = outputs[-1][0]
        elif isinstance(self.model, rt.capi.onnxruntime_inference_collection.InferenceSession):
            blob = cv2.dnn.blobFromImage(image, self.rescale, self.input_shape)
            outputs = self.model.run([], {self.input_name : blob})
            # first model layer is of shape (1, 25500, 85)
            outputs = outputs[0][0]
        boxes, confidences, class_ids = self._detect(outputs, image)     
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.5)
        boxes, class_ids = ([boxes[i] for i in indexes], [class_ids[i] for i in indexes])
        return boxes, class_ids

    # only works for outputs of shape (25500, 85)
    def _detect(self, outputs, image):
        image_width, image_height, _ = image.shape
        x_factor = image_width / 640
        y_factor =  image_height / 640
        outputs = [row for row in outputs if np.amax(row[5:])>0.5 and row[4]>0.5]
        confidences, class_ids = ([row[4] for row in outputs], [np.argmax(row[5:]) for row in outputs])
        xywh = [(row[0], row[1], row[2], row[3]) for row in outputs]
        boxes = [np.array([int((x - 0.5 * w) * x_factor), int((y - 0.5 * h) * y_factor), int(w * x_factor), int(h * y_factor)])
        for x, y, w, h in xywh]
        return boxes, confidences, class_ids


if __name__=='__main__':
    # need cudnn and CUDA toolkit installed for cuda as ONNXRuntime backend
    detector = Detector('yolov5x6.onnx', 1/255.0, (640, 640), backend='cuda')
    image = cv2.imread('assets/detect.jpg')
    start = time.time()
    boxes, class_ids = detector.detect(image)
    class_list = []
    with open("coco.names", "r") as f:
        class_list = [cname.strip() for cname in f.readlines()]
    for i in range(len(class_ids)):
        box = boxes[i]
        class_id = class_ids[i]
        # overlay rectangle boxes and class name 
        cv2.rectangle(image, box, (0, 255, 255), 2)
        cv2.rectangle(image, (box[0], box[1] - 20), (box[0] + box[2], box[1]), (0, 255, 255), -1)
        cv2.putText(image, class_list[class_id], (box[0], box[1] - 40), cv2.FONT_ITALIC, 4, (255,0,0), 5)
    cv2.imwrite('assets/inference.jpg', image)
    print('YOLO took', time.time() - start, 'seconds')
