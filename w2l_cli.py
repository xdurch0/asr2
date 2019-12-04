import argparse
from w2l.main import run_asr


parser = argparse.ArgumentParser(description="See README.md")
parser.add_argument("mode",
                    choices=["train", "predict", "eval-current", "eval-all",
                             "return", "errors"],
                    help="What to do. 'train', 'predict', 'eval-current', "
                         "'eval-all', 'return' or 'errors'.")
parser.add_argument("data_config",
                    help="Path to data config file. See code for details.")
parser.add_argument("model_dir",
                    help="Path to store checkpoints etc.")

parser.add_argument("-f", "--data_format",
                    default="channels_first",
                    choices=["channels_first", "channels_last"],
                    help="Data format. Either 'channels_first' "
                         "(default, recommended for GPU) "
                         "or 'channels_last', necessary for CPU.")
parser.add_argument("-c", "--cpu",
                    action="store_true",
                    help="Set this flag when running on a CPU (dear god). This "
                         "will slightly change how transcriptions are handled "
                         "in training. No need to pass this if you are not "
                         "training. Note that CPU should also use "
                         "channels_last data format (always, not just when "
                         "training). NOTE: "
                         "Currently, this flag is basically always in effect "
                         "because the CTC 'GPU mode' seems to be much slower "
                         "than the CPU one even when running on GPU. So you "
                         "can pass it, or not. It might do something again in "
                         "the future.")
parser.add_argument("-r", "--reg",
                    nargs=2,
                    default=[None, "0.0"],
                    metavar=["reg_type", "reg_coeff"],
                    help="Which regularizer to use and the coefficient. The "
                         "type is multiple things separated by underscores:"
                         "target_norm_edges_nhsize. Target should be weight or "
                         "act depending on whether to apply the regularizer to "
                         "the filter weights or the feature maps. norm is "
                         "which distance measure to use: l1, l2, linf or cos. "
                         "edges is how to treat edges: no (no special treatment), "
                         "occ (weight inversely to occurrences) or wrap (wrap "
                         "around the grid). nhsize is how big the neighborhoods "
                         "should be. Uneven numbers >= 3! Default: "
                         "No regularization.")

parser.add_argument("-A", "--adam_params",
                    nargs=4,
                    type=float,
                    default=[1e-4, 0.9, 0.9, 1e-8],
                    metavar=["adam_lr", "adam_beta1", "adam_beta2" "adam_eps"],
                    help="Learning rate, beta1 and beta2 and epsilon for "
                         "Adam. Defaults: 1e-4, 0.9, 0.9, 1e-8.")
parser.add_argument("-B", "--batch_size",
                    type=int,
                    default=16,  # small but seems to work well
                    help="Batch size. Default: 16.")
parser.add_argument("-C", "--clipping",
                    type=float,
                    default=500.0,
                    help="Global norm to clip gradients to. Default: 500. "
                         "If no clipping is desired, pass 0 here.")
parser.add_argument("-F", "--fix_lr",
                    action="store_true",
                    help="Set this flag to use the LR given as adam_params "
                         "as-is. If this is not set, it will be decayed "
                         "automatically when training progress seems to halt. "
                         "NOTE: The decaying process will still happen with "
                         "this flag set -- it just won't have an effect. "
                         "However should you restart training without this "
                         "flag, you will get whatever learning rate the "
                         "decaying process has reached at that time.")
parser.add_argument("-N", "--normalize_off",
                    action="store_true",
                    help="Pass this to *disable* data normalization. If this "
                         "is not given, input arrays will be individually "
                         "scaled to mean 0 and std 1. Keep in mind that the "
                         "data may already have been normalized in "
                         "preprocessing. Check the corresponding data config.")
parser.add_argument("-S", "--steps",
                    type=int,
                    default=250000,
                    help="Number of training steps to take. Default: 250000. "
                         "Ignored if doing prediction or evaluation.")
parser.add_argument("-T", "--threshold",
                    type=float,
                    default=0.,
                    help="Threshold to clip small input values. Any values "
                         "more than this much under the maximum will be "
                         "clipped. E.g. if the max is 15 and the threshold is "
                         "50, any value below -35 would be clipped to -35. It "
                         "is your responsibility to pass a reasonable value "
                         "here -- this can vary heavily depending on the "
                         "scale of the data. Passing 0 or any 'False' value "
                         "here disables thresholding. NOTE: You probably "
                         "don't want to use this with pre-normalized data "
                         "since in that case, each example is essentially on "
                         "its own scale (one that results in mean 0 and std "
                         "1, or whatever normalization was used) so a single "
                         "threshold value isn't really applicable. However, "
                         "it is perfectly fine to use this with the -N flag "
                         "off, since that normalization will be performed "
                         "*after* thresholding. Default: 0, disables "
                         "thresholding.")
parser.add_argument("-W", "--which_sets",
                    default="",
                    help="Which data subsets to use. Pass as comma-separated "
                         "string. If not given, everything will be used!!")
args = parser.parse_args()

if args.which_sets:
    which_sets = args.which_sets.split(",")
else:
    which_sets = None

out = run_asr(mode=args.mode, data_config=args.data_config,
              model_dir=args.model_dir,
              data_format=args.data_format, cpu=args.cpu,
              adam_params=args.adam_params, batch_size=args.batch_size,
              clipping=args.clipping, fix_lr=args.fix_lr,
              normalize=not args.normalize_off, steps=args.steps,
              threshold=args.threshold, which_sets=which_sets)
