import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import time
from detect import *


class Detector:
    def __init__(self, args):
        self.args = args
        self.enlarge = args.enlarge
        self.load_model()

    def load_model(self):
        torch.set_grad_enabled(False)
        self.cfg = None
        if self.args.network == "mobile0.25":
            self.cfg = cfg_mnet
        elif self.args.network == "resnet50":
            self.cfg = cfg_re50
        # net and model
        net = RetinaFace(cfg=self.cfg, phase = 'test')
        net = load_model(net, self.args.trained_model, self.args.cpu)
        net.eval()
        print('Finished loading model!')
        print(net)
        cudnn.benchmark = True
        self.device = torch.device("cpu" if self.args.cpu else "cuda")
        self.net = net.to(self.device)    

    def detect(self, img_raw, resize=1, verbose=False):
        img = np.float32(img_raw)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        tic = time.time()
        loc, conf, landms = self.net(img)  # forward pass

        if verbose:
            print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.args.nms_threshold)
        # keep = nms(dets, self.args.nms_threshold,force_cpu=self.args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.args.keep_top_k, :]
        landms = landms[:self.args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)

        return dets

    def annotate(self, img_raw, dets):
        # annotate image
        for b in dets:
            if b[4] < self.args.vis_thres:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(img_raw, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # landms
            cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)            

        return img_raw

    def run(self):
        cap = cv2.VideoCapture(0)

        # font which we will be using to display FPS 
        font = cv2.FONT_HERSHEY_SIMPLEX 

        # used to record the time when we processed last frame 
        prev_frame_time = 0

        # used to record the time at which we processed current frame 
        new_frame_time = 0

        if not cap.isOpened():
            print('Failed to open video')
            return
        
        while cap.isOpened():
            ret, img = cap.read()
            self.height, self.width = img.shape[:2]
            while ret:
                dets = self.detect(img)
                img = self.annotate(img, dets)

                # time when we finish processing for this frame 
                new_frame_time = time.time() 

                # Calculating the fps 

                # fps will be number of frame processed in given time frame 
                # since their will be most of time error of 0.001 second 
                # we will be subtracting it to get more accurate result 
                fps = 1/(new_frame_time-prev_frame_time) 
                prev_frame_time = new_frame_time 

                # converting the fps into integer 
                fps = int(fps) 

                # converting the fps to string so that we can display it on frame 
                # by using putText function 
                fps = "FPS: " + str(fps) 

                # putting the FPS count on the frame 
                cv2.putText(img, fps, (10, 40), font, 1, (100, 255, 0), 3, cv2.LINE_AA) 

                cv2.imshow('Live', img)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

                ret, img = cap.read()

            cap.release()
            cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description='Retinaface')

    parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
    parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
    parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
    parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
    parser.add_argument('--enlarge', default=0, type=int, help='enlarge size of each detection window dimension')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    det = Detector(args)
    det.run()

