from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os
import argparse

import qtorch
import fixed_quant
import static_quant
import auto_low
from qtorch.auto_low import sequential_lower
from qtorch import FloatingPoint
from qtorch.quant import float_quantize

parser = argparse.ArgumentParser(
    prog='python3 detection.py',
    description='Run mtcnn on lfw dataset',
    )

parser.add_argument('-q', dest='quantize_method', type=str, default=None,
                    help='use quantized model')
parser.add_argument('-s', dest='save', action='store_true',
                    help='save model')

if __name__ == '__main__':
    args = parser.parse_args()

    workers = 0 if os.name == 'nt' else 4

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )

    def collate_fn(x):
        return x[0]

    dataset = datasets.ImageFolder('lfw/')
    dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

    if args.quantize_method == 'dymatic_quant_int8':
        print("Using dymatic_quant_int8 model")
        mtcnn = auto_low.sequential_lower(mtcnn,
                                          fixed_quant.Quantizer,
                                          layer_types=['conv','linear'],
                                          device=device)

        weight_quantizer = lambda x: fixed_quant.QG(x)
        for name, param in mtcnn.named_parameters():
            param.data = weight_quantizer(
                param.data
            )
    elif args.quantize_method == 'static_quant_int8':
        print("Using static_quant_int8 model")
        mtcnn = auto_low.sequential_lower(mtcnn,
                                          static_quant.Quantizer,
                                          layer_types=['conv','linear'],
                                          device=device)

        weight_quantizer = lambda x: fixed_quant.QG(x)
        for name, param in mtcnn.named_parameters():
            param.data = weight_quantizer(
                param.data
            )

        count = 0
        for x, y in loader:
            x_aligned, prob = mtcnn(x, return_prob=True)
            count += 1
            if (count > 100):
                break
        static_quant.lower(mtcnn)
        static_quant.show(mtcnn)
    elif args.quantize_method == 'float_quant':
        print("Using float_quant model")
        forward_num = FloatingPoint(exp=5, man=3)
        mtcnn = sequential_lower(mtcnn,
                                 layer_types=['conv','linear'],
                                 forward_number=forward_num,
                                 forward_rounding="nearest")

        weight_quantizer = lambda x : float_quantize(x, exp=5, man=3, rounding="nearest")
        for name, param in mtcnn.named_parameters():
            param.data = weight_quantizer(
                param.data
            )
    else:
        print("Using original model")

    print(mtcnn)

    aligned = []
    names = []
    probs = []
    count = 0
    for x, y in loader:
        x_aligned, prob = mtcnn(x, return_prob=True)
        probs.append(prob)
        if x_aligned is not None:
            print('Face detected with probability: {:8f}'.format(prob))
            aligned.append(x_aligned)
            names.append(dataset.idx_to_class[y])

    if args.save == True:
        torch.save(mtcnn.state_dict(), "model")

    print('Overall probability: {:8f}'.format(sum(probs) / len(probs)) )
