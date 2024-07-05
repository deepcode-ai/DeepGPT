'''
Copyright 2024 The Deepcode-AI DeepGPT Team
'''

from deepgpt.pt.deepgpt_light import DeepGPTLight
from deepgpt.pt.deepgpt_light import ADAM_OPTIMIZER, LAMB_OPTIMIZER
from deepgpt.pt.deepgpt_lr_schedules import add_tuning_arguments

try:
    from deepgpt.version_info import git_hash, git_branch
except ImportError:
    git_hash = None
    git_branch = None

__version__ = 0.1
__git_hash__ = git_hash
__git_branch__ = git_branch


def initialize(args,
               model,
               optimizer=None,
               model_parameters=None,
               training_data=None,
               lr_scheduler=None,
               mpu=None,
               dist_init_required=True,
               collate_fn=None):
    r"""Initialize the DeepGPT Engine.

    Arguments:
        args: a dictionary containing local_rank and deepgpt_config
            file location

        model: Required: nn.module class before apply any wrappers

        optimizer: Optional: a user defined optimizer, this is typically used instead of defining
            an optimizer in the DeepGPT json config.

        model_parameters: Optional: An iterable of torch.Tensors or dicts.
            Specifies what Tensors should be optimized.

        training_data: Optional: Dataset of type torch.utils.data.Dataset

        lr_scheduler: Optional: Learning Rate Scheduler Object. It should define a get_lr(),
            step(), state_dict(), and load_state_dict() methods

        mpu: Optional: A model parallelism unit object that implements
            get_model/data_parallel_group/rank/size()

        dist_init_required: Optional: Initializes torch.distributed

        collate_fn: Optional: Merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.

    Return:
        The following tuple is returned by this function.
        tuple: engine, engine.optimizer, engine.training_dataloader, engine.lr_scheduler

        engine: DeepGPT runtime engine which wraps the client model for distributed training.

        engine.optimizer: Wrapped optimizer if a user defined optimizer is passed or
            if optimizer is specified in json config else None.

        engine.training_dataloader: DeepGPT dataloader if training data was passed else None.

        engine.lr_scheduler: Wrapped lr scheduler if user lr scheduler is passed
            or if lr scheduler specified in json config else None.


    """
    print("DeepGPT info: version={}, git-hash={}, git-branch={}".format(
        __version__,
        __git_hash__,
        __git_branch__),
          flush=True)

    engine = DeepGPTLight(args=args,
                            model=model,
                            optimizer=optimizer,
                            model_parameters=model_parameters,
                            training_data=training_data,
                            lr_scheduler=lr_scheduler,
                            mpu=mpu,
                            dist_init_required=dist_init_required,
                            collate_fn=collate_fn)

    return_items = [
        engine,
        engine.optimizer,
        engine.training_dataloader,
        engine.lr_scheduler
    ]
    return tuple(return_items)


def _add_core_arguments(parser):
    r"""Helper (internal) function to update an argument parser with an argument group of the core DeepGPT arguments.
        The core set of DeepGPT arguments include the following:
        1) --deepgpt: boolean flag to enable DeepGPT
        2) --deepgpt_config <json file path>: path of a json configuration file to configure DeepGPT runtime.

        This is a helper function to the public add_config_arguments()

    Arguments:
        parser: argument parser
    Return:
        parser: Updated Parser
    """
    group = parser.add_argument_group('DeepGPT', 'DeepGPT configurations')

    group.add_argument('--deepgpt',
                       default=False,
                       action='store_true',
                       help='Enable DeepGPT')

    group.add_argument('--deepgpt_config',
                       default=None,
                       type=str,
                       help='DeepGPT json configuration file.')

    return parser


def add_config_arguments(parser):
    r"""Update the argument parser to enabling parsing of DeepGPT command line arguments.
        The set of DeepGPT arguments include the following:
        1) --deepgpt: boolean flag to enable DeepGPT
        2) --deepgpt_config <json file path>: path of a json configuration file to configure DeepGPT runtime.

    Arguments:
        parser: argument parser
    Return:
        parser: Updated Parser
    """
    parser = _add_core_arguments(parser)

    return parser
