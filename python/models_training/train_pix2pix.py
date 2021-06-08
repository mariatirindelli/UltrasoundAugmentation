import configargparse
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from utils.CustomWandbLogger import CustomWandbLogger
from utils.utils import (
    argparse_summary,
    get_class_by_path,
)
from utils.configargparse_arguments import build_configargparser
from datetime import datetime

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import os
from polyaxon_client.tracking import Experiment
from modules.cut_module import CUT2D

def train_pix2pix(hparams, ModuleClass, ModelClass, DataModuleClass, DatasetClass, logger):
    """
    Main training routine specific for this project
    :param hparams:
    """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    # load model
    if ModelClass is None:
        model = None
    else:
        model = ModelClass(hparams=hparams)

    dataset = DataModuleClass(hparams=hparams, dataset=DatasetClass)
    module = ModuleClass(hparams=hparams, model=model, logger=logger)

    # ------------------------
    # 3 INIT TRAINER --> continues training
    # ------------------------
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{hparams.output_path}/checkpoints/",
        period=hparams.save_every_k_epochs,  # Interval (number of epochs) between checkpoints.
        verbose=True,
        prefix='ckpt',
        save_top_k=-1,  # if set to -1 all models are saved every <period> epochs
        filename=f'{{epoch}}'
    )

    trainer = Trainer(
        gpus=hparams.gpus,
        logger=logger,
        fast_dev_run=hparams.fast_dev_run,
        min_epochs=hparams.min_epochs,
        max_epochs=hparams.max_epochs,
        checkpoint_callback=True,
        resume_from_checkpoint=hparams.resume_from_checkpoint,
        callbacks=[checkpoint_callback],  # [early_stop_callback, checkpoint_callback]
        weights_summary='full',
        num_sanity_val_steps=hparams.num_sanity_val_steps,
        log_every_n_steps=hparams.log_every_n_steps,
        auto_lr_find=True,
        auto_scale_batch_size=True,
        #limit_train_batches=0.01,  # use 0.2 for Polyaxon, use 0.03 to avoid memory error on Anna's computer
        #limit_val_batches=0.01,  # use 0.4 for Polyaxon, use 0.05 to avoid memory error on Anna's computer
    )
    # ------------------------
    # 4 START TRAINING
    # ------------------------

    dataset.prepare_data()

    if isinstance(module, CUT2D):
        module.data_dependent_initialize(next(iter(dataset.train_dataloader())))

    if not hparams.test_only:

        trainer.fit(module, train_dataloader=dataset.train_dataloader(), val_dataloaders=dataset.val_dataloader())

        if len(dataset.test_dataloader()) == 0:
            print("No test data available")
            return
        trainer.test()
    else:
        print(
            f"Best: {checkpoint_callback.best_model_score} | monitor: {checkpoint_callback.monitor} "
            f"| path: {checkpoint_callback.best_model_path}"
            f"\nTesting..."
        )
        # For the pix2pix network  we process the training data with the GAN as it "belongs" to the training process
        print("resuming checkpoint from: {}".format(hparams.resume_from_checkpoint))
        dataset.prepare_data()

        if len(dataset.test_dataloader()) == 0:
            print("No test data available")
            return

        trainer.test(ckpt_path=hparams.resume_from_checkpoint, model=module, test_dataloaders=dataset.test_dataloader())

    print("test done")


if __name__ == "__main__":
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments

    root_dir = Path(__file__).parent
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('-c', is_config_file=True, help='config file path')
    parser, hparams = build_configargparser(parser)

    # each LightningModule defines arguments relevant to it
    # ------------------------
    # LOAD MODULE
    # ------------------------
    module_path = f"modules.{hparams.module}"
    ModuleClass = get_class_by_path(module_path)
    parser = ModuleClass.add_module_specific_args(parser)
    # ------------------------
    # LOAD MODEL
    # ------------------------
    # TODO: for gans add generator and discriminator models

    if hparams.model == "":
        ModelClass = None
    else:
        model_path = f"models.{hparams.model}"
        ModelClass = get_class_by_path(model_path)
        parser = ModelClass.add_model_specific_args(parser)
    # ------------------------
    # LOAD DATAMODULE
    # ------------------------
    datamodule_path = f"datamodules.{hparams.datamodule}"
    DataModuleClass = get_class_by_path(datamodule_path)
    parser = DataModuleClass.add_dataset_specific_args(parser)
    # ------------------------
    # LOAD DATASET
    # ------------------------
    dataset_path = f"datasets.{hparams.dataset}"
    DatasetClass = get_class_by_path(dataset_path)

    # ------------------------
    #  PRINT PARAMS & INIT LOGGER
    # ------------------------
    hparams = parser.parse_args()
    # hparams.data_root = input_folder
    # print(hparams.data_root)
    # hparams.augmentation_prob = augmentation_prob

    # setup logging
    exp_name = (
            hparams.module.split(".")[-1]
            + "_"
            + hparams.dataset.split(".")[-1]
            + "_"
            + hparams.model.replace(".", "_")
    )
    if hparams.on_polyaxon:
        experiment = Experiment()
        hparams.output_path = Path(experiment.get_outputs_path())
        hparams.name = experiment.experiment_id + "_" + exp_name

    else:
        date_str = datetime.now().strftime("%y%m%d-%H%M%S_")
        hparams.name = date_str + exp_name
        hparams.output_path = Path(hparams.output_path).absolute() / hparams.name
        if not os.path.exists(hparams.output_path):
            os.mkdir(hparams.output_path)

    argparse_summary(hparams, parser)
    logger = CustomWandbLogger(hparams)

    # ---------------------
    # RUN TRAINING
    # ---------------------

    train_pix2pix(hparams, ModuleClass, ModelClass, DataModuleClass, DatasetClass, logger)
