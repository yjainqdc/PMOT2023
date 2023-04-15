import torch
import numpy as np

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device


class Detector:

    def __init__(self):
        self.img_size = 640
        self.threshold = 0.4
        self.stride = 1

        self.weights = './yolov7small.pt'

        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        model = attempt_load(self.weights, map_location=self.device)
        model.to(self.device).eval()
        # model.half()

        self.m = model
        self.names = model.module.names if hasattr(
            model, 'module') else model.names

    def preprocess(self, img):

        shape = img.shape
        img = letterbox(img, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        # img = img.half()
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return shape, img

    def detect(self, im):

        shape, img = self.preprocess(im)

        pred = self.m(img, augment=False)[0]
        pred = pred.float()
        pred = non_max_suppression(pred, self.threshold, 0.3)

        boxes = []
        boxes_tensor = []
        for det in pred:

            if det is not None and len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], shape).round()

                for *x, conf, cls_id in det:
                    lbl = self.names[int(cls_id)]
                    # if lbl not in ['fuyou']:
                    if lbl not in ['ceratium furca', 'ceratium fucus', 'ceratium trichoceros',
                                'chaetoceros curvisetus', 'cladocera', 'copepoda1','copepoda2',
                                'coscinodiscus','curve thalassiosira','guinardia delicatulad','helicotheca',
                                'lauderia cleve','skeletonema','thalassionema nitzschioides',
                                'thalassiosira nordenskioldi','tintinnid','sanguinea','Thalassiosira rotula','Protoperidinium','Eucampia zoodiacus','Guinardia striata']:
                        continue
                    pass
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    boxes.append(
                        (x1, y1, x2, y2, lbl, conf))
                    boxes_tensor.append(
                        [x1, y1, x2-x1, y2-y1, conf,float(cls_id)])
        boxes_tensor = torch.Tensor(boxes_tensor)
        print(boxes_tensor)
        return boxes,boxes_tensor
