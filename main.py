import os
import argparse
import datetime
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent
for _key, _sub in (('TORCH_HOME', 'torch'), ('XDG_CACHE_HOME', 'xdg'), ('MPLCONFIGDIR', 'mpl')):
    os.environ.setdefault(_key, str(CODE_ROOT / '.cache' / _sub))

import torch

from util import *
from models import *


STAGES = ['split', 'verify', 'probe', 'tune', 'train', 'infer', 'manyway', 'perturb', 'profile', 'qualitative', 'evaluate', 'report']
SEEDS = [13, 29, 47, 61, 79, 97, 113, 131, 149, 167]


def get_args_parser():
    parser = argparse.ArgumentParser('Rival-aware partial transport for few-shot fine-grained image classification', add_help=False)
    parser.add_argument('--stage', default='all', choices=['all'] + STAGES, help='pipeline stage to run (default: all, in order)')
    parser.add_argument('--data-root', default=str(CODE_ROOT / 'data'), type=str,
                        help='folder holding CUB_200_2011/, StanfordDogs/ and StanfordCars/')
    parser.add_argument('--cub-path', default='', type=str, help='CUB-200-2011 root (images.txt, bounding_boxes.txt, images/)')
    parser.add_argument('--dogs-path', default='', type=str, help='Stanford Dogs root (Images/, Annotation/)')
    parser.add_argument('--cars-path', default='', type=str, help='Stanford Cars root (cars_train/, cars_test/, devkit .mat files)')
    parser.add_argument('--output_dir', default=str(CODE_ROOT / 'results' / 'full'), type=str,
                        help='path where all intermediate and final outputs are stored')
    parser.add_argument('--device', default='cuda', help='device to use')
    parser.add_argument('--seed', default=0, type=int, help='seed of the fixed evaluation manifests and tuning subsets')
    parser.add_argument('--seeds', default=SEEDS, type=int, nargs='+', help='paired training seeds')
    parser.add_argument('--overwrite', action='store_true', help='recompute cached outputs of the selected stages')

    parser.add_argument('--datasets', default=list(DATASETS), nargs='+', choices=list(DATASETS))
    parser.add_argument('--shots', default=[1, 5], type=int, nargs='+')
    parser.add_argument('--experiments', default=list(EXPERIMENTS), nargs='+', choices=list(EXPERIMENTS),
                        help='primary benchmark, V_conf confirmation, coverage sweep, kappa diagnostic, ResNet-12 panel')
    parser.add_argument('--methods', default=[], nargs='*', choices=list(METHODS), help='restrict training/inference to these methods')
    parser.add_argument('--split-seed', default=0, type=int, help='class-split seed for Stanford Dogs/Cars and the CUB V_tune/V_conf pools')
    parser.add_argument('--split-dir', default='', type=str, help='optional folder with {dataset}_split.csv (class_id, split)')
    parser.add_argument('--image-size', default=84, type=int, help='raw uncropped images are resized to this square input')

    parser.add_argument('--n-way', default=5, type=int)
    parser.add_argument('--train-episodes', default=10000, type=int)
    parser.add_argument('--train-queries', default=5, type=int, help='queries per class during training')
    parser.add_argument('--val-every', default=500, type=int, help='checkpoint selection interval on V_tune (episodes)')
    parser.add_argument('--val-episodes', default=1000, type=int, help='fixed V_tune episodes used for checkpoint selection')
    parser.add_argument('--test-episodes', default=2000, type=int)
    parser.add_argument('--test-queries', default=15, type=int, help='queries per class during evaluation')
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--lr-milestones', default=[0.6, 0.85], type=float, nargs='+', help='fractions of the budget with a tenfold decay')
    parser.add_argument('--lr-gamma', default=0.1, type=float)
    parser.add_argument('--jitter', default=0.4, type=float, help='brightness/contrast/saturation jitter strength')
    parser.add_argument('--tau-init', default=10.0, type=float, help='initial learnable logit scale')

    parser.add_argument('--rho', default=0.6, type=float, help='common transported-mass budget')
    parser.add_argument('--entropy', default=0.05, type=float, help='entropy weight epsilon')
    parser.add_argument('--lambda-g', default=0.25, type=float, help='row-bias weight of the gate')
    parser.add_argument('--eta', default=0.1, type=float, help='gate temperature')
    parser.add_argument('--t-m', default=0.1, type=float, help='soft-minimum temperature of the rival pools')
    parser.add_argument('--kappa', default=0.0, type=float, help='residual-dispersion interpolation (primary: 0)')
    parser.add_argument('--descriptor-floor', default=1e-6, type=float)
    parser.add_argument('--shared-offset', default=0.0, type=float, help='calibration offset b_s of shared gates')
    parser.add_argument('--shared-temp', default=0.1, type=float, help='temperature eta_s of shared gates')
    parser.add_argument('--weight-temp', default=0.1, type=float, help='softmax temperature t_w of weighted pooling')
    parser.add_argument('--candidate-offset', default=0.5, type=float, help='offset b of the candidate-only gate C1')
    parser.add_argument('--candidate-temp', default=0.1, type=float, help='temperature eta_+ of the candidate-only gate C1')
    parser.add_argument('--attn-temp', default=0.1, type=float, help='temperature of dense softmax attention (A0)')
    parser.add_argument('--toprow-temp', default=0.1, type=float, help='temperature t_attn of top-row matching (C3)')
    parser.add_argument('--anchor-alpha', default=0.25, type=float, help='candidate-specific share of the anchored bias')
    parser.add_argument('--permutation-seed', default=0, type=int, help='fixed label-independent permutation of the C4 gate')
    parser.add_argument('--emd-entropy', default=0.05, type=float, help='Sinkhorn entropy of the DeepEMD adaptation')
    parser.add_argument('--dn4-neighbours', default=3, type=int)
    parser.add_argument('--sfmr-radius', default=0.5, type=float, help='normalized low-pass radius of the SFMR frequency split')
    parser.add_argument('--ot-tol', default=1e-4, type=float, help='normalized plan-change tolerance')
    parser.add_argument('--ot-feas-tol', default=1e-5, type=float, help='mass and capacity residual tolerance')
    parser.add_argument('--ot-max-iter', default=500, type=int, help='sweep cap')
    parser.add_argument('--ot-patience', default=2, type=int, help='consecutive sweeps meeting both criteria')
    parser.add_argument('--ot-fallback-iter', default=5000, type=int, help='inference-time continuation for solves that reach the cap')

    parser.add_argument('--use-tuned', action='store_true', help='apply the validation-selected configuration from the tune stage')
    parser.add_argument('--tune-dataset', default='cub', choices=list(DATASETS))
    parser.add_argument('--tune-shot', default=1, type=int)
    parser.add_argument('--tune-seed', default=SEEDS[0], type=int)
    parser.add_argument('--tune-episodes', default=10000, type=int)
    parser.add_argument('--tune-trials', default=12, type=int, help='gate trials per gated control')
    parser.add_argument('--tune-methods', default=list(TUNED_METHODS), nargs='+', choices=list(METHODS))
    parser.add_argument('--entropy-grid', default=[0.02, 0.05, 0.1], type=float, nargs='+')
    parser.add_argument('--lambda-grid', default=[0.1, 0.25, 0.5], type=float, nargs='+')
    parser.add_argument('--eta-grid', default=[0.05, 0.1, 0.2, 0.5], type=float, nargs='+')
    parser.add_argument('--tm-grid', default=[0.03, 0.1, 0.2], type=float, nargs='+')
    parser.add_argument('--offset-grid', default=[0.0, 0.02, 0.05], type=float, nargs='+')
    parser.add_argument('--shared-temp-grid', default=[0.05, 0.1, 0.2], type=float, nargs='+')
    parser.add_argument('--weight-temp-grid', default=[0.05, 0.1, 0.2], type=float, nargs='+')
    parser.add_argument('--candidate-offset-grid', default=[0.3, 0.5, 0.7], type=float, nargs='+')
    parser.add_argument('--attn-temp-grid', default=[0.05, 0.1, 0.2], type=float, nargs='+')

    parser.add_argument('--coverage-grid', default=[0.4, 0.6, 0.8, 1.0], type=float, nargs='+')
    parser.add_argument('--kappa-grid', default=[0.0, 0.25, 0.5, 0.75, 1.0], type=float, nargs='+')
    parser.add_argument('--kappa-retrain', default=0.25, type=float)
    parser.add_argument('--small-entropy', default=0.02, type=float, help='smaller-entropy A3 control at kappa=0')
    parser.add_argument('--ways', default=[5, 10, 20, 50], type=int, nargs='+')
    parser.add_argument('--manyway-episodes', default=2000, type=int)
    parser.add_argument('--followup-methods', default=list(FOLLOWUP_METHODS), nargs='+', choices=list(METHODS))
    parser.add_argument('--blur-sigmas', default=[0.5, 1.0, 1.5], type=float, nargs=3)
    parser.add_argument('--jpeg-qualities', default=[30, 15, 8], type=int, nargs=3)
    parser.add_argument('--brightness-shifts', default=[0.1, 0.2, 0.3], type=float, nargs=3)
    parser.add_argument('--contrast-factors', default=[0.7, 0.5, 0.3], type=float, nargs=3)
    parser.add_argument('--noise-sigmas', default=[0.04, 0.08, 0.12], type=float, nargs=3)
    parser.add_argument('--background-weights', default=[1 / 3, 2 / 3, 1.0], type=float, nargs=3)

    parser.add_argument('--profile-warmup', default=100, type=int)
    parser.add_argument('--profile-passes', default=5, type=int)
    parser.add_argument('--profile-episodes', default=20, type=int, help='episodes per interleaved pass and seed block')
    parser.add_argument('--qual-size', default=160, type=int, help='square input of the ResNet-50 reference features (5x5 grid)')
    parser.add_argument('--qual-episodes', default=300, type=int)

    parser.add_argument('--probe-seed', default=20261104, type=int, help='seed of the procedural descriptor banks')
    parser.add_argument('--easy-separation', default=0.55, type=float)
    parser.add_argument('--easy-noise', default=0.35, type=float)
    parser.add_argument('--hard-separation', default=0.20, type=float)
    parser.add_argument('--hard-noise', default=0.60, type=float)
    parser.add_argument('--n-boot', default=50000, type=int, help='episode-cluster bootstrap resamples of the ambiguity analysis')
    parser.add_argument('--participation-threshold', default=0.5, type=float, help='row participates above this fraction of its capacity')
    parser.add_argument('--practical-margin', default=0.3, type=float, help='working practical-benefit margin (pp)')
    parser.add_argument('--latency-budget', default=5.0, type=float, help='maximum additional forward latency (%%)')
    parser.add_argument('--memory-budget', default=2.0, type=float, help='maximum additional peak memory (%%)')
    parser.add_argument('--margin-grid', default=[0.0, 0.1, 0.3, 0.5, 1.0], type=float, nargs='+')
    return parser


def main(args):
    print(args)
    torch.manual_seed(args.seed)
    ws = Workspace(args.output_dir)
    runners = {
        'split': stage_split, 'verify': stage_verify, 'probe': stage_probe, 'tune': stage_tune, 'train': stage_train, 'infer': stage_infer,
        'manyway': stage_manyway, 'perturb': stage_perturb, 'profile': stage_profile, 'qualitative': stage_qualitative,
        'evaluate': stage_evaluate, 'report': stage_report,
    }
    stages = STAGES if args.stage == 'all' else [args.stage]
    if args.stage == 'all' and not args.use_tuned:
        stages = [s for s in stages if s != 'tune']
        utils.log('tune is skipped: the paper configuration is used (pass --use-tuned to run validation-only tuning and apply it)')
    start_time = time.time()
    for stage in stages:
        stage_start = time.time()
        utils.log(f'stage {stage}: start')
        runners[stage](args, ws)
        utils.log(f'stage {stage}: done in {datetime.timedelta(seconds=int(time.time() - stage_start))}')
    total_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print('Total time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Rival-aware partial transport for few-shot fine-grained image classification', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
