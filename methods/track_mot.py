import numpy as np

# 通过这里切换模型
from detector import Detector
import cv2
from tracker.basetrack import BaseTracker, STrack  # for framework
from tracker.deepsort import DeepSORT
from tracker.bytetrack import ByteTrack
from tracker.deepmot import DeepMOT
from tracker.botsort import BoTSORT
from tracker.uavmot import UAVMOT
from tracker.strongsort import StrongSORT
import argparse
from draw import draw_bboxes

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default=r'E:\DATASETGOGOGOGOGO\scriptPMOT\PMOT2022\test\DENSE_FAST\img\video.avi', help='video path')

    parser.add_argument('--tracker', type=str, default='bytetrack', help='sort, deepsort, etc')
    parser.add_argument('--model_path', type=str, default='../weights/yolov7.pt', help='model path')
    parser.add_argument('--img_size', nargs='+', type=int, default=[720,1080], help='[train, test] image sizes')

    """For tracker"""
    # model path
    parser.add_argument('--reid_model_path', type=str, default='./weights/ckpt.t7', help='path for reid model path')
    parser.add_argument('--dhn_path', type=str, default='./weights/DHN.pth', help='path of DHN path for DeepMOT')

    # threshs
    parser.add_argument('--conf_thresh', type=float, default=0.5, help='filter tracks')
    parser.add_argument('--nms_thresh', type=float, default=0.7, help='thresh for NMS')
    parser.add_argument('--iou_thresh', type=float, default=0.5, help='IOU thresh to filter tracks')

    # other options
    parser.add_argument('--track_buffer', type=int, default=30, help='tracking buffer')
    parser.add_argument('--gamma', type=float, default=0.1, help='param to control fusing motion and apperance dist')
    parser.add_argument('--kalman_format', type=str, default='default',
                        help='use what kind of Kalman, default, naive, strongsort or bot-sort like')
    parser.add_argument('--min_area', type=float, default=150, help='use to filter small bboxs')


    opts = parser.parse_args()














    '''超参数，选择模型'''
    TRACKER_DICT = {
        'sort': BaseTracker,
        'deepsort': DeepSORT,
        'bytetrack': ByteTrack,
        'deepmot': DeepMOT,
        'botsort': BoTSORT,
        'uavmot': UAVMOT,
        'strongsort': StrongSORT,
    }  # dict for trackers, key: str, value: class(BaseTracker)

    tracker = TRACKER_DICT[opts.tracker](opts, frame_rate=30, gamma=opts.gamma)

    # 写入result.txt获取类别用
    cls_list_index = ['ceratium furca', 'ceratium fucus', 'ceratium trichoceros',
'chaetoceros curvisetus', 'cladocera', 'copepoda1','copepoda2',
'coscinodiscus','curve thalassiosira','guinardia delicatulad','helicotheca',
'lauderia cleve','skeletonema','thalassionema nitzschioides',
'thalassiosira nordenskioldi','tintinnid','sanguinea','Thalassiosira rotula','Protoperidinium','Eucampia zoodiacus','Guinardia striata']

    # 初始化2个撞线polygon

    # 初始化 yolov5
    detector = Detector()

    # 打开视频
    # capture = cv2.VideoCapture(r'E:\DATASETGOGOGOGOGO\scriptPMOT\PMOT2022\val\DENSE_FAST\img\video.avi')
    capture = cv2.VideoCapture(opts.video)

    # result.txt
    file_handle = open('result_m_deepsort.txt', mode='w')

    frame = 0
    #轨迹尾巴
    taillist = [[] for i in range(1000)]
    while True:
        # 读取每帧图片
        _, im = capture.read()
        if im is None:
            break

        # 缩小尺寸，4096*3000 -> 1080*720
        scale_x = 4096 / 1080
        scale_y = 3000 / 720
        im = cv2.resize(im, (1080, 720))


        bboxes,boxes_tensor = detector.detect(im)
        print('-----------------------')
        print(bboxes)
        # 如果画面中 有bbox

        if len(bboxes) > 0:
            STrack_list = tracker.update(boxes_tensor, im)
        if len(bboxes) > 0 and len(STrack_list)>0:
            list_bboxs = []
            #处理跟踪框
            for trk in STrack_list:
                bbox = trk.tlwh
                bbox[:2] += bbox[2:] // 2
                bbox[2] = bbox[0] + bbox[2]
                bbox[3] = bbox[1] + bbox[3]
                id = trk.track_id
                cls = trk.cls
                list_bboxs.append((int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]),cls_list_index[int(cls.item())],id))


            list = []
            for i in list_bboxs:
                w = i[2] - i[0]
                h = i[3] - i[1]
                if i[4] != '':
                    template = "%d,%d,%d,%d,%d,%d,1,%d,1\n" % (
                    frame, i[5], i[0] * scale_x, i[1] * scale_y, w * scale_x, h * scale_y, cls_list_index.index(i[4]))
                    # print(template)
                    list.append(template)
            file_handle.writelines(list)

            # list_bboxs = bboxes
            # 画框
            print('++++++++++++++++++++++++++++++++++++')
            print(list_bboxs)

            #画轨迹
            for (x1, y1, x2, y2, cls_id, pos_id) in list_bboxs:
                temp = (int((x1 + x2) / 2),int((y1 + y2) / 2))
                taillist[int(pos_id)].append(temp)
                if len(taillist[int(pos_id)]) > 25:
                    taillist[int(pos_id)].pop(0)
            output_image_frame = draw_bboxes(im, list_bboxs, line_thickness=None,list = taillist)

            pass
        else:
            # 如果画面中 没有bbox
            output_image_frame = im
        pass

        # 保存图片
        # cv2.imwrite(r'C:\Users\qwer\Desktop\\111\\'+str(frame)+'.jpg', output_image_frame)
        cv2.imshow('demo', output_image_frame)
        # cv2.imshow(output_image_frame)
        cv2.waitKey(1)

        frame += 1
        pass
    pass
    file_handle.close()
    capture.release()
    cv2.destroyAllWindows()
