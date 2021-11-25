import configargparse
from distutils.util import strtobool

def regnet_argparse(config_files="regnet.ini"):
    p = configargparse.ArgParser(default_config_files=[config_files])
    p.add_argument('--config-files', required=False, is_config_file=True, help='path of configure file')

    # ORGANIZATION
    p.add_argument('--model-dir', default='saved_models', help='model output directory')
    p.add_argument('--log-dir', default='logs', help='log output directory')
    p.add_argument('--data-dir', help='path or glob pattern of data files')
    p.add_argument('--device', default='cuda', help='running device: cpu or cuda')
    p.add_argument('--use-last-ckpt', type=strtobool, default=False, help='whether to use last checkerpoint')
    p.add_argument('--load-model', default=None, help='path to the last checkerpoint file')
    p.add_argument('--start-epoch', type=int, default=0, help='start epoch if use last checkerpoint')
    p.add_argument('--eval-best-epoch', type=int, default=10, help='when to start saving best model')

    # NETWORK
    p.add_argument('--int-steps', type=int, default=0, help='number of integration steps (default: 0)')
    p.add_argument('--int-downsize', type=int, default=2,
                   help='the flow downsample factor for vector integration  (default: 2)')
    p.add_argument('--bidir', type=strtobool, default=False, help='whether to do bidirectional registration')
    p.add_argument('--num-channel-initial', type=int, default=32, help='number of initial channels of RegNet')
    p.add_argument('--extract-levels', type=int, nargs='+', default=[0, 1, 2, 3],
                   help='which level(s) is/are extracted to output DVF/DDF')

    # TRAINING
    p.add_argument('--num-samples', type=int, default=1,
                   help='number of samples from random crop in dataset processing')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--max-epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--inshape', type=int, nargs='+', default=[64, 128, 128],
                   help='list of the spatial dimensions of the input')
    p.add_argument('--val-interval', type=int, default=5, help='interval of validations')
    p.add_argument('--save-interval', type=int, default=50, help='interval of model save')
    p.add_argument('--decay-rate', type=float, default=1.0, help='decay rate of learning rate shceduler')
    p.add_argument('--flipping', type=strtobool, default=False, help='whether to random flip axes of image')

    # LOSS
    p.add_argument('--ncc-loss-weight', type=float, default=1.0, help='weight of normalized cross correlation loss')
    p.add_argument('--mind-loss-weight', type=float, default=1.0, help='weight of MINDSCC loss')
    p.add_argument('--dice-loss-weight', type=float, default=1.0, help='weight of dice loss')
    p.add_argument('--grad-loss-weight', type=float, default=0.1, help='weight of gradient loss of flow field')
    p.add_argument('--ngf-loss-weight', type=float, default=1.0, help='weight of normalized gradient fields')

    arg = p.parse_args()
    return arg