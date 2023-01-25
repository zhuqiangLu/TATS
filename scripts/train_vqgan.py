# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tats import VQGAN, VideoData, DataModuleFromConfig
from tats.modules.callbacks import ImageLogger, VideoLogger

def main():
    pl.seed_everything(1234)

    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = VQGAN.add_model_specific_args(parser)

    parser = argparse.ArgumentParser(parents=[parser], add_help=False)
    parser.add_argument('--data_path', type=str, default='/datasets01/Kinetics400_Frames/videos')
    parser.add_argument('--sequence_length', type=int, default=16)
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--image_channels', type=int, default=3)
    parser.add_argument('--smap_cond', type=int, default=0)
    parser.add_argument('--smap_only', action='store_true')
    parser.add_argument('--text_cond', action='store_true')
    parser.add_argument('--vtokens', action='store_true')
    parser.add_argument('--vtokens_pos', action='store_true')
    parser.add_argument('--spatial_length', type=int, default=15)
    parser.add_argument('--sample_every_n_frames', type=int, default=1)
    parser.add_argument('--image_folder', action='store_true')
    parser.add_argument('--stft_data', action='store_true')
    # parser = VideoData.add_data_specific_args(parser)

    # data = VideoData(args)
    args = parser.parse_args()

    train_data_config = {
        'target': 'tats.VideoDataset',
        'params':
            {
                'folder': '/root/autodl-tmp/train_split_frame',
                'image_width': 480,
                'image_height': 272,
                'cube_size': args.resolution,
                'clip_size': args.sequence_length,
            }
    }
    val_data_config = {
        'target': 'tats.VideoDataset',
        'params':
            {
                'folder': '/root/autodl-tmp/test_split_frame',
                'image_width': 480,
                'image_height': 272,
                'cube_size': args.resolution,
                'clip_size': args.sequence_length,
            }
    }
    


    data = DataModuleFromConfig(
        batch_size = args.batch_size,
        train = train_data_config,
        validation = val_data_config,
        num_workers = args.num_workers,
    )

    # pre-make relevant cached files if necessary
    data.prepare_data()
    # data.train_dataloader()
    # data.test_dataloader()
    

    # automatically adjust learning rate
    bs, base_lr, ngpu, accumulate = args.batch_size, args.lr, args.gpus, args.accumulate_grad_batches
    args.lr = accumulate * (ngpu/8.) * (bs/4.) * base_lr
    print("Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus/8) * {} (batchsize/4) * {:.2e} (base_lr)".format(
        args.lr, accumulate, ngpu/8, bs/4, base_lr))

    model = VQGAN(args)

    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='val/recon_loss', save_top_k=3, mode='min', filename='latest_checkpoint'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=3000, save_top_k=-1, filename='{epoch}-{step}-{train/recon_loss:.2f}'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=10000, save_top_k=-1, filename='{epoch}-{step}-10000-{train/recon_loss:.2f}'))
    # callbacks.append(ImageLogger(batch_frequency=750, max_images=4, clamp=True))
    callbacks.append(VideoLogger(batch_frequency=1500, max_videos=4, clamp=True))

    kwargs = dict()
    if args.gpus > 1:
        kwargs = dict(distributed_backend='ddp', gpus=args.gpus)

    # load the most recent checkpoint file
    base_dir = os.path.join(args.default_root_dir, 'lightning_logs')
    if os.path.exists(base_dir):
        log_folder = ckpt_file = ''
        version_id_used = step_used = 0
        for folder in os.listdir(base_dir):
            version_id = int(folder.split('_')[1])
            if version_id > version_id_used:
                version_id_used = version_id
                log_folder = folder
        if len(log_folder) > 0:
            ckpt_folder = os.path.join(base_dir, log_folder, 'checkpoints')
            if os.path.exists(ckpt_folder):
                for fn in os.listdir(ckpt_folder):
                    if fn == 'latest_checkpoint.ckpt':
                        ckpt_file = 'latest_checkpoint_prev.ckpt'
                        os.rename(os.path.join(ckpt_folder, fn), os.path.join(ckpt_folder, ckpt_file))
                if len(ckpt_file) > 0:
                    args.resume_from_checkpoint = os.path.join(ckpt_folder, ckpt_file)
                    print('will start from the recent ckpt %s'%args.resume_from_checkpoint)

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks,  strategy = "ddp",
                                            max_steps=args.max_steps, precision=16, amp_backend='native', **kwargs)

    trainer.fit(model, data)


if __name__ == '__main__':
    main()

