from io import BytesIO

import PIL.Image
import numpy as np

from pydarknet import Detector, Image


def test_image():
    img = PIL.Image.open(BytesIO('darknet_cfg/test_images/test-f17-09-345.jpg'))

    img = np.array(img)
    img = img[:,:,::-1] # RGB to BGR

    net = Detector(bytes("darknet_cfg/yolov3-tiny.cfg", encoding="utf-8"), 
                   bytes("darknet_cfg/yolov3-tiny_10000.weights", encoding="utf-8"), 0,
                   bytes("darknet_cfg/voc.names", encoding="utf-8"))

    img2 = Image(img)

    results = net.detect(img2)

    results_labels = [x[0].decode("utf-8") for x in results]

    assert "bicycle" in results_labels
    assert "dog" in results_labels
    assert "truck" in results_labels
    assert len(results_labels) == 3