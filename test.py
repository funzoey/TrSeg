import argparse
from isegm.utils.exp_imports.default import *
from torch.utils.data import DataLoader
from isegm.model.modeling.transformer_helper.cross_entropy_loss import CrossEntropyLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=55,
                        metavar='N', help='Dataloader threads.')

    parser.add_argument('--workers', type=int, default=0,
                        metavar='N', help='Dataloader threads.')

    parser.add_argument('--bs', type=int, default=2,
                        help='You can override model batch size by specify positive number.')

    parser.add_argument("--save_dir", type=str, default='./checkpoint')

    parser.add_argument("--log_dir", type=str, default='./log')

    parser.add_argument("--data_dir", type=str, default='./datasets')

    return parser.parse_args()

def prepare_data(args):
    testset = OAIZIBDataset(
        args.data_dir,
        split='val',
    )

    cdataloader = DataLoader(
        testset, args.bs,
        drop_last=True, pin_memory=True,
        num_workers=args.workers
    )
    return cdataloader

def init_model():
    backbone = dict(
        in_chans=3,
        in_coord_chans=2,
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=False,
    )

    head = dict(
        in_channels=[192, 384, 768, 1536],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=1,
        loss_decode=CrossEntropyLoss(),
        align_corners=False,
    )

    model = SwinformerModel(
        backbone_params=backbone,
        head_params=head,
        use_naive_concat=False,
        use_rgb_conv=False,
        use_deep_fusion=True,
        use_disks=True,
        norm_radius=5,
        with_prev_mask=False,
    )
    return model

def main():
    args = parse_args()
    cloader = prepare_data(args)
    model = init_model().to(device)

    for epoch in range(args.epochs):
        for datas in cloader:
            batch_data = {k: v.to(device) for k, v in datas.items()}
            output = model(batch_data['images'], batch_data['points'])
            print(output)

if __name__ == '__main__':
    main()