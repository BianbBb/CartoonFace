import torch
import torch.nn as nn
import sys
sys.path.append("..")
import config as cfg
from utils.util import _gather_feat, _transpose_and_gather_feat
from backbone.Hourglass.large_hourglass import HourglassNet




def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def ctdet_decode(heat, wh, reg=None, cat_spec_wh=False, K=100):# TODO: 查看一下ctdet_decode 的输出
    batch, cat, height, width = heat.size()

    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)

    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
        reg = _transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _transpose_and_gather_feat(wh, inds)
    if cat_spec_wh:
        wh = wh.view(batch, K, cat, 2)
        clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
        wh = wh.gather(2, clses_ind).view(batch, K, 2)
    else:
        wh = wh.view(batch, K, 2)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)

    return detections

#
# def run(self, image_or_path_or_tensor, meta=None):
#     load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
#     merge_time, tot_time = 0, 0
#     debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug == 3),
#                         theme=self.opt.debugger_theme)
#     start_time = time.time()
#     pre_processed = False
#     if isinstance(image_or_path_or_tensor, np.ndarray):
#         image = image_or_path_or_tensor
#     elif type(image_or_path_or_tensor) == type(''):
#         image = cv2.imread(image_or_path_or_tensor)
#     else:
#         image = image_or_path_or_tensor['image'][0].numpy()
#         pre_processed_images = image_or_path_or_tensor
#         pre_processed = True
#
#     loaded_time = time.time()
#     load_time += (loaded_time - start_time)
#
#     detections = []
#     for scale in self.scales:
#         scale_start_time = time.time()
#         if not pre_processed:
#             images, meta = self.pre_process(image, scale, meta)
#         else:
#             # import pdb; pdb.set_trace()
#             images = pre_processed_images['images'][scale][0]
#             meta = pre_processed_images['meta'][scale]
#             meta = {k: v.numpy()[0] for k, v in meta.items()}
#         images = images.to(self.opt.device)
#         torch.cuda.synchronize()
#         pre_process_time = time.time()
#         pre_time += pre_process_time - scale_start_time
#
#         output, dets, forward_time = self.process(images, return_time=True)
#
#         torch.cuda.synchronize()
#         net_time += forward_time - pre_process_time
#         decode_time = time.time()
#         dec_time += decode_time - forward_time
#
#         if self.opt.debug >= 2:
#             self.debug(debugger, images, dets, output, scale)
#
#         dets = self.post_process(dets, meta, scale)
#         torch.cuda.synchronize()
#         post_process_time = time.time()
#         post_time += post_process_time - decode_time
#
#         detections.append(dets)
#
#     results = self.merge_outputs(detections)
#     torch.cuda.synchronize()
#     end_time = time.time()
#     merge_time += end_time - post_process_time
#     tot_time += end_time - start_time
#
#     if self.opt.debug >= 1:
#         self.show_results(debugger, image, results)
#
#     return {'results': results, 'tot': tot_time, 'load': load_time,
#             'pre': pre_time, 'net': net_time, 'dec': dec_time,
#             'post': post_time, 'merge': merge_time}


def main(para):

    logger = para.logger
    # device = para.device

    logger.debug('------ Load Network ------')
    net = HourglassNet(para.heads, num_stacks=1, num_branchs=para.num_branchs)
    try:
        net.load_state_dict(torch.load(para.exp_path))
        logger.info('Net Parameters Loaded Successfully!')
    except FileNotFoundError:
        logger.warning('Can not find feature.pkl !')
        return False

    logger.debug('------ Test ------')

    # 读取数据
    # 按train进行归一化，修改
    # 输入网络
    # 输出
    # decode
    # cv2显示结果


if __name__ == '__main__':
    parameter = cfg.Detection_Parameter()
    main(parameter)
