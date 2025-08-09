import os
from easydict import EasyDict as edict
# loss function related
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss, MSELoss, CrossEntropyLoss
# train pipeline related
from lib.train.trainers import LTRTrainer
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
from lib.models.seqtrack import build_seqtrack
from lib.train.actors import SeqTrackActor
# phase configuration related
from lib.config.seqtrack.training_phases import Phase
import importlib

def run(settings):
    settings.description = 'Training script for SeqTrack'

    # First, verify the config file exists
    if not os.path.exists(settings.cfg_file):
        raise FileNotFoundError(f"Config file not found: {settings.cfg_file}")
    
    # Initialize Phase object with the config file path from settings
    settings.phase_manager = Phase(config_path=settings.cfg_file)
    
    # Load and update configuration
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg  # generate cfg from lib.config
    config_module.update_config_from_file(settings.cfg_file)  # update cfg from experiments
    
    # Update settings with the loaded config
    update_settings(settings, cfg)
    
    # Print configuration (excluding PHASES as it's now handled by Phase class)
    if settings.local_rank in [-1, 0]:
        print("\n" + "="*80)
        print("Training Configuration:")
        print("="*80)
        for key in cfg.keys():
            if key != 'PHASES':  # Skip PHASES as it's now handled by Phase class
                print(f"{key}:")
                print(f"  {cfg[key]}")
        print("="*80 + "\n")
    
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders
    loader_type = getattr(cfg.DATA, "LOADER", "tracking")
    if loader_type == "tracking":
        loader_train = build_dataloaders(cfg, settings)
    else:
        raise ValueError("illegal DATA LOADER")

    # Create network
    if settings.script_name == "seqtrack":
        net = build_seqtrack(cfg)
    else:
        raise ValueError("illegal script name")

    # wrap networks to distributed one
    net.cuda()
    if settings.local_rank != -1:
        net = DDP(net, broadcast_buffers=False, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")
    # Loss functions and Actors
    if settings.script_name == "seqtrack":
        bins = cfg.MODEL.BINS
        weight = torch.ones(bins + 2)
        weight[bins] = 0.01
        weight[bins + 1] = 0.01
        objective = {'ce': CrossEntropyLoss(weight=weight)}
        loss_weight = {'ce': cfg.TRAIN.CE_WEIGHT}
        actor = SeqTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)

    else:
        raise ValueError("illegal script name")

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    trainer = LTRTrainer(actor, [loader_train], optimizer, settings, lr_scheduler, use_amp=use_amp)
    trainer.train(cfg.TRAIN.max_epochs, load_latest=True, fail_safe=True)
    #trainer.train(cfg.TRAIN.max_epochs, load_latest=False, load_ckpt=5, fail_safe=True)
