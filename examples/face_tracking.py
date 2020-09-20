from facenet_pytorch import MTCNN
import torch
import numpy as np
import mmcv, cv2
from PIL import Image, ImageDraw
from IPython import display
import argparse

from qtorch.auto_low import sequential_lower
from qtorch import FloatingPoint
from qtorch.quant import float_quantize
import qtorch
import fixed_quant
import static_quant
import auto_low


parser = argparse.ArgumentParser(
    prog='python3 detection.py',
    description='Run mtcnn on lfw dataset',
    )

parser.add_argument('-q', dest='quantize_method', type=str, default=None,
                    help='use quantized model')
parser.add_argument('-load', dest='load', action='store_true',
                    help='save model')

if __name__ == '__main__':
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    mtcnn = MTCNN(keep_all=True, device=device)

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

        mtcnn.load_state_dict(torch.load("model"))
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

    video = mmcv.VideoReader('video.mp4')
    frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]

    print(len(frames))

    display.Video('video.mp4', width=640)

    frames_tracked = []
    for i, frame in enumerate(frames):
        print('\rTracking frame: {}'.format(i + 1), end='')
        
        # Detect faces
        boxes, _ = mtcnn.detect(frame)
        
        # Draw faces
        frame_draw = frame.copy()
        draw = ImageDraw.Draw(frame_draw)
        if(boxes is not None):
            for box in boxes:
                draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
        
        # Add to frame list
        # frames_tracked.append(frame_draw.resize((640, 360), Image.BILINEAR))
        frames_tracked.append(frame_draw)
    print('\nDone')

    dim = frames_tracked[0].size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video_tracked = cv2.VideoWriter('video_tracked.mp4', fourcc, 25.0, dim)
    for frame in frames_tracked:
        video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    video_tracked.release()

