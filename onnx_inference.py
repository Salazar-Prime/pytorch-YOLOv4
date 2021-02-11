import sys
import onnx
import os
import argparse
import numpy as np
import cv2
import onnxruntime

from tool.utils import *
from tool.darknet2onnx import *


def main(onnx_file, image_path, name_file, batch_size):

    session = onnxruntime.InferenceSession(onnx_file)
    #session = onnx.load(onnx_file)
    print("The model expects input shape: ", session.get_inputs()[0].shape)

    image_src = cv2.imread(image_path)
    
    # change based on model
    ttime = 0
    iterations = 20
    
    for i in range(iterations+10):
        boxes, t = detect(session, name_file, image_src)
            
        # total time for 10 iterations - skip first Because the first iteration is usually longer
        if i<10:
            continue
        ttime = ttime + t 
    
    class_names = load_class_names(name_file)
    plot_boxes_cv2(image_src, boxes[0], savename='predictions_onnx.jpg', class_names=class_names)
    
    print('-----------------------------------')
    print('    ONNX inference time: %f' % (ttime/iterations))
    print('    Frames per second:   %0.2f' % (iterations/ttime))
    print('-----------------------------------')





def detect(session, name_file, image_src):
    IN_IMAGE_H = session.get_inputs()[0].shape[2]
    IN_IMAGE_W = session.get_inputs()[0].shape[3]
    #IN_IMAGE_W=416
    #IN_IMAGE_H=416
    
    ta = time.time()
    
    # Input
    resized = cv2.resize(image_src, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0
    print("Shape of the network input: ", img_in.shape)

    # Compute
    input_name = session.get_inputs()[0].name

    outputs = session.run(None, {input_name: img_in})
    
    tb = time.time()
    
    boxes = post_processing(img_in, 0.4, 0.6, outputs)
    
    print('-----------------------------------')
    print('    ONNX inference time: %f' % (tb-ta))
    print('-----------------------------------')
    
    return boxes, (tb-ta)
    
    

if __name__ == '__main__':
    if len(sys.argv) == 5:
        onnx_file = sys.argv[1]
        image_path = sys.argv[2]
        name_file = sys.argv[3]
        batch_size = int(sys.argv[4])
        main(onnx_file, image_path, name_file, batch_size)
    else:
        print('Please run this way:\n')
        print('  python demo_onnx.py <onnxFile> <imageFile> <name_file> <batchSize>')
