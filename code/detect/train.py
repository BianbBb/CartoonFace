
import cv2
import os
import torch
import torch.utils.data as DATA
import sys
sys.path.append("..")
import config as cfg
from detect.dataload import DetDataset
from backbone.Hourglass.large_hourglass import HourglassNet
from detect.trainer import DetTrainer

def main(para):
    torch.manual_seed(para.seed)
    torch.backends.cudnn.benchmark = True

    logger = para.logger
    #device = para.device

    logger.debug('------ Load Network ------')
    net = HourglassNet(para.heads, num_stacks=1, num_branchs=para.num_branchs)
    # # from torchsummary import summary
    # #     # summary(net.cuda(),(3,512,512),batch_size=8)
    # #     # print(net)

    logger.debug('------ Load Dataset ------')
    Train_Data = DetDataset(para, flag='train', train_num=para.train_num)
    train_loader = DATA.DataLoader(dataset=Train_Data, batch_size=para.BATCH_SIZE * len(para.gpu_ids), shuffle=True, drop_last=True)

    Val_Data = DetDataset(para, flag='validation', train_num=para.train_num)
    val_loader = DATA.DataLoader(dataset=Val_Data, batch_size=para.BATCH_SIZE * len(para.gpu_ids), shuffle=False, drop_last=True)

    # for step, data in enumerate(train_loader):
    #     logger.debug(step)
    #     logger.debug(data['reg_mask'])
    #
    # logger.debug('--------')
    #
    # for step, data in enumerate(val_loader):
    #     logger.debug(step)
    #     logger.debug(data['reg_mask'])

    logger.debug('------     Train    ------')
    Trainer = DetTrainer(para, net, train_loader=train_loader, val_loader=val_loader, optimizer='SGD')
    Trainer.run()
    # start_epoch = 0
    #
    # if para.resume:
    #     if os.path.isfile(para.exp_path):
    #         try:
    #             net.load_state_dict(torch.load(para.exp_path))
    #             logger.debug('Net Parameters Loaded Successfully')
    #         except FileNotFoundError:
    #             logger.debug('Please check pkl file name')
    #     else:
    #         logger.debug('Can not find pkl file')
    #
    # Trainer = DetTrainer(para, net, optimizer)
    #
    # Trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device) ###
    #
    # Trainer.run()
    # print('Setting up data...')
    # val_loader = torch.utils.data.DataLoader(
    #     Dataset(opt, 'val'),
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=1,
    #     pin_memory=True
    # )
    #
    # if opt.test:
    #     _, preds = trainer.val(0, val_loader)
    #     val_loader.dataset.run_eval(preds, opt.save_dir)
    #     return
    #
    # train_loader = torch.utils.data.DataLoader(
    #     Dataset(opt, 'train'),
    #     batch_size=opt.batch_size,
    #     shuffle=True,
    #     num_workers=opt.num_workers,
    #     pin_memory=True,
    #     drop_last=True
    # )
    #
    # print('Starting training...')
    # best = 1e10
    # for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    #     mark = epoch if opt.save_all else 'last'
    #     log_dict_train, _ = trainer.train(epoch, train_loader)
    #
    #     for k, v in log_dict_train.items():
    #         logger.scalar_summary('train_{}'.format(k), v, epoch)
    #         logger.write('{} {:8f} | '.format(k, v))
    #     if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
    #         # save_Path 以epoch命名
    #         torch.save(net.state_dict(),save_path)
    #
    #         with torch.no_grad():
    #             log_dict_val, preds = trainer.val(epoch, val_loader)
    #         for k, v in log_dict_val.items():
    #             logger.scalar_summary('val_{}'.format(k), v, epoch)
    #             logger.write('{} {:8f} | '.format(k, v))
    #         if log_dict_val[opt.metric] < best:
    #             best = log_dict_val[opt.metric]
    #             save_model(os.path.join(opt.save_dir, 'model_best.pth'),
    #                        epoch, model)
    #     else:
    #         save_model(os.path.join(opt.save_dir, 'model_last.pth'),
    #                    epoch, model, optimizer)
    #     logger.write('\n')
    #     if epoch in opt.lr_step:
    #         save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
    #                    epoch, model, optimizer)
    #         lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
    #         print('Drop LR to', lr)
    #         for param_group in optimizer.param_groups:
    #             param_group['lr'] = lr
    # logger.close()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    parameter = cfg.Detection_Parameter()
    main(parameter)
