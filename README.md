# Tubes-Viskom

from mmdet3d.apis import init_model, inference_detector, show_result_meshlab
config_file = '../configs/second/hv_second_secfpn_6x8_80e_kitti-3d-car.py'

# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = '../work_dirs/second/epoch_40.pth'

# build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device='cuda:0')

# test a single sample
pcd = 'kitti_000008.bin'
result, data = inference_detector(model, pcd)

# show the results
out_dir = './'
show_result_meshlab(data, result, out_dir)

from argparse import ArgumentParser

from mmdet3d.apis import (inference_mono_3d_detector, init_model,
                          show_result_meshlab)


def main():
    parser = ArgumentParser()
    parser.add_argument('image', help='image file')
    parser.add_argument('ann', help='ann file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.15, help='bbox score threshold')
    parser.add_argument(
        '--out-dir', type=str, default='demo', help='dir to save results')
    parser.add_argument(
        '--show',
        action='store_true',
        help='show online visualization results')
    parser.add_argument(
        '--snapshot',
        action='store_true',
        help='whether to save online visualization results')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    # test a single image
    result, data = inference_mono_3d_detector(model, args.image, args.ann)
    # show the results
    show_result_meshlab(
        data,
        result,
        args.out_dir,
        args.score_thr,
        show=args.show,
        snapshot=args.snapshot,
        task='mono-det')


if __name__ == '__main__':
    main()
    
    from argparse import ArgumentParser

from mmdet3d.apis import (inference_multi_modality_detector, init_model,
                          show_result_meshlab)


def main():
    parser = ArgumentParser()
    parser.add_argument('pcd', help='Point cloud file')
    parser.add_argument('image', help='image file')
    parser.add_argument('ann', help='ann file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.0, help='bbox score threshold')
    parser.add_argument(
        '--out-dir', type=str, default='demo', help='dir to save results')
    parser.add_argument(
        '--show',
        action='store_true',
        help='show online visualization results')
    parser.add_argument(
        '--snapshot',
        action='store_true',
        help='whether to save online visualization results')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    # test a single image
    result, data = inference_multi_modality_detector(model, args.pcd,
                                                     args.image, args.ann)
    # show the results
    show_result_meshlab(
        data,
        result,
        args.out_dir,
        args.score_thr,
        show=args.show,
        snapshot=args.snapshot,
        task='multi_modality-det')


if __name__ == '__main__':
    main()
    
    from argparse import ArgumentParser

from mmdet3d.apis import inference_segmentor, init_model, show_result_meshlab


def main():
    parser = ArgumentParser()
    parser.add_argument('pcd', help='Point cloud file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--out-dir', type=str, default='demo', help='dir to save results')
    parser.add_argument(
        '--show',
        action='store_true',
        help='show online visualization results')
    parser.add_argument(
        '--snapshot',
        action='store_true',
        help='whether to save online visualization results')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    # test a single image
    result, data = inference_segmentor(model, args.pcd)
    # show the results
    show_result_meshlab(
        data,
        result,
        args.out_dir,
        show=args.show,
        snapshot=args.snapshot,
        task='seg',
        palette=model.PALETTE)


if __name__ == '__main__':
    main()
