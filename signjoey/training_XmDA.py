#!/usr/bin/env python
def add_project_root(): #@jinhui
    import sys
    from os.path import abspath, join, dirname
    sys.path.insert(0, abspath(join(abspath(dirname(__file__)), '../')))

add_project_root()

import torch, random

torch.backends.cudnn.deterministic = True
from signjoey import metrics
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import argparse
import numpy as np
import os
import shutil
import time
import queue
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from signjoey.model import build_model
from signjoey.batch import Batch, Batch_Sign, Batch_gls2text
from signjoey.helpers import (
    log_data_info,
    load_config,
    log_cfg,
    load_checkpoint,
    make_model_dir,
    make_logger,
    set_seed,
    symlink_update,
)
from signjoey.model import SignModel, SignMixupModel
from signjoey.prediction_XmDA import validate_on_data
from signjoey.loss import XentLoss
from signjoey.data import load_data, make_data_iter
from signjoey.builders import build_optimizer, build_scheduler, build_gradient_clipper
from signjoey.prediction_XmDA import test
from signjoey.metrics import wer_single
from signjoey.vocabulary import SIL_TOKEN
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchtext.data import Dataset
from typing import List, Dict
from signjoey.model import build_model_mixup


# pylint: disable=too-many-instance-attributes
class TrainManager:
    """ Manages training loop, validations, learning rate scheduling
    and early stopping."""

    def __init__(self, model: SignMixupModel, config: dict) -> None:
        """
        Creates a new TrainManager for a model, specified as in configuration.

        :param model: torch module defining the model
        :param config: dictionary containing the training configurations
        """
        train_config = config["training"]
        self.config = config
        # files for logging and storing
        self.model_dir = make_model_dir(
            model_dir=train_config["model_dir"], overwrite=train_config.get("overwrite", False)
        )
        self.logger = make_logger(model_dir=self.model_dir)
        self.logging_freq = train_config.get("logging_freq", 100)
        self.valid_report_file = "{}/validations.txt".format(self.model_dir)
        self.tb_writer = SummaryWriter(log_dir=self.model_dir + "/tensorboard/")

        # input
        self.feature_size = (
            sum(config["data"]["feature_size"])
            if isinstance(config["data"]["feature_size"], list)
            else config["data"]["feature_size"]
        )
        self.dataset_version = config["data"].get("version", "phoenix_2014_trans")

        # model
        self.model = model
        self.txt_pad_index = self.model.txt_pad_index
        self.txt_bos_index = self.model.txt_bos_index
        self._log_parameters_list()
        # Check if we are doing only recognition or only translation or both
        self.do_recognition = (
            config["training"].get("recognition_loss_weight", 1.0) > 0.0
        )
        self.do_translation = (
            config["training"].get("translation_loss_weight", 1.0) > 0.0
        )

        # Get Recognition and Translation specific parameters
        if self.do_recognition:
            self._get_recognition_params(train_config=train_config)
        if self.do_translation:
            self._get_translation_params(train_config=train_config)

        # optimization
        self.last_best_lr = train_config.get("learning_rate", -1)
        self.learning_rate_min = train_config.get("learning_rate_min", 1.0e-8)
        self.clip_grad_fun = build_gradient_clipper(config=train_config)
        self.optimizer = build_optimizer(
            config=train_config, parameters=model.parameters()
        )
        self.batch_multiplier = train_config.get("batch_multiplier", 1)

        # validation & early stopping
        self.validation_freq = train_config.get("validation_freq", 100)
        self.num_valid_log = train_config.get("num_valid_log", 5)
        self.ckpt_queue = queue.Queue(maxsize=train_config.get("keep_last_ckpts", 5))
        self.eval_metric = train_config.get("eval_metric", "bleu")
        if self.eval_metric not in ["bleu", "chrf", "wer", "rouge"]:
            raise ValueError(
                "Invalid setting for 'eval_metric': {}".format(self.eval_metric)
            )
        self.early_stopping_metric = train_config.get(
            "early_stopping_metric", "eval_metric"
        )

        # if we schedule after BLEU/chrf, we want to maximize it, else minimize
        # early_stopping_metric decides on how to find the early stopping point:
        # ckpts are written when there's a new high/low score for this metric
        if self.early_stopping_metric in [
            "ppl",
            "translation_loss",
            "recognition_loss",
        ]:
            self.minimize_metric = True
        elif self.early_stopping_metric == "eval_metric":
            if self.eval_metric in ["bleu", "chrf", "rouge"]:
                assert self.do_translation
                self.minimize_metric = False
            else:  # eval metric that has to get minimized (not yet implemented)
                self.minimize_metric = True
        else:
            raise ValueError(
                "Invalid setting for 'early_stopping_metric': {}".format(
                    self.early_stopping_metric
                )
            )

        # data_augmentation parametersdel trainer
        self.frame_subsampling_ratio = config["data"].get(
            "frame_subsampling_ratio", None
        )
        self.random_frame_subsampling = config["data"].get(
            "random_frame_subsampling", None
        )
        self.random_frame_masking_ratio = config["data"].get(
            "random_frame_masking_ratio", None
        )

        # learning rate scheduling
        self.scheduler, self.scheduler_step_at = build_scheduler(
            config=train_config,
            scheduler_mode="min" if self.minimize_metric else "max",
            optimizer=self.optimizer,
            hidden_size=config["model"]["encoder"]["hidden_size"],
        )

        # data & batch handling
        self.level = config["data"]["level"]
        if self.level not in ["word", "bpe", "char"]:
            raise ValueError("Invalid segmentation level': {}".format(self.level))

        self.shuffle = train_config.get("shuffle", True)
        self.epochs = train_config["epochs"]
        self.batch_size = train_config["batch_size"]
        self.batch_type = train_config.get("batch_type", "sentence")
        self.eval_batch_size = train_config.get("eval_batch_size", self.batch_size)
        self.eval_batch_type = train_config.get("eval_batch_type", self.batch_type)

        self.use_cuda = train_config["use_cuda"]
        if self.use_cuda:
            self.model.cuda()
            if self.do_translation:
                self.translation_loss_function.cuda()
            if self.do_recognition:
                self.recognition_loss_function.cuda()

        # initialize training statistics
        self.steps = 0
        # stop training if this flag is True by reaching learning rate minimum
        self.stop = False
        self.total_txt_tokens = 0
        self.total_gls_tokens = 0
        self.best_ckpt_iteration = 0
        # initial values for best scores
        self.best_ckpt_score = np.inf if self.minimize_metric else -np.inf
        self.best_all_ckpt_scores = {}
        # comparison function for scores
        self.is_best = (
            lambda score: score < self.best_ckpt_score
            if self.minimize_metric
            else score > self.best_ckpt_score
        )

        # model parameters
        if "load_model" in train_config.keys():
            model_load_path = train_config["load_model"]
            self.logger.info("Loading model from %s", model_load_path)
            reset_best_ckpt = train_config.get("reset_best_ckpt", False)
            reset_scheduler = train_config.get("reset_scheduler", False)
            reset_optimizer = train_config.get("reset_optimizer", False)
            self.init_from_checkpoint(
                model_load_path,
                reset_best_ckpt=reset_best_ckpt,
                reset_scheduler=reset_scheduler,
                reset_optimizer=reset_optimizer,
            )

    def _get_recognition_params(self, train_config) -> None:
        # NOTE (Cihan): The blank label is the silence index in the gloss vocabulary.
        #   There is an assertion in the GlossVocabulary class's __init__.
        #   This is necessary to do TensorFlow decoding, as it is hardcoded
        #   Currently it is hardcoded as 0.
        self.gls_silence_token = self.model.gls_vocab.stoi[SIL_TOKEN]
        assert self.gls_silence_token == 0

        self.recognition_loss_function = torch.nn.CTCLoss(
            blank=self.gls_silence_token, zero_infinity=True
        )
        self.recognition_loss_weight = train_config.get("recognition_loss_weight", 1.0)
        self.eval_recognition_beam_size = train_config.get(
            "eval_recognition_beam_size", 1
        )

    def _get_translation_params(self, train_config) -> None:
        self.label_smoothing = train_config.get("label_smoothing", 0.0)
        self.translation_loss_function = XentLoss(
            pad_index=self.txt_pad_index, smoothing=self.label_smoothing
        )
        self.translation_normalization_mode = train_config.get(
            "translation_normalization", "batch"
        )
        if self.translation_normalization_mode not in ["batch", "tokens"]:
            raise ValueError(
                "Invalid normalization {}.".format(self.translation_normalization_mode)
            )
        self.translation_loss_weight = train_config.get("translation_loss_weight", 1.0)
        self.eval_translation_beam_size = train_config.get(
            "eval_translation_beam_size", 1
        )
        self.eval_translation_beam_alpha = train_config.get(
            "eval_translation_beam_alpha", -1
        )
        self.translation_max_output_length = train_config.get(
            "translation_max_output_length", None
        )

    def _save_checkpoint(self) -> None:
        """
        Save the model's current parameters and the training state to a
        checkpoint.

        The training state contains the total number of training steps,
        the total number of training tokens,
        the best checkpoint score and iteration so far,
        and optimizer and scheduler states.

        """
        model_path = "{}/{}.ckpt".format(self.model_dir, self.steps)
        state = {
            "steps": self.steps,
            "total_txt_tokens": self.total_txt_tokens if self.do_translation else 0,
            "total_gls_tokens": self.total_gls_tokens if self.do_recognition else 0,
            "best_ckpt_score": self.best_ckpt_score,
            "best_all_ckpt_scores": self.best_all_ckpt_scores,
            "best_ckpt_iteration": self.best_ckpt_iteration,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict()
            if self.scheduler is not None
            else None,
        }
        torch.save(state, model_path)
        if self.ckpt_queue.full():
            to_delete = self.ckpt_queue.get()  # delete oldest ckpt
            try:
                os.remove(to_delete)
            except FileNotFoundError:
                self.logger.warning(
                    "Wanted to delete old checkpoint %s but " "file does not exist.",
                    to_delete,
                )

        self.ckpt_queue.put(model_path)

        # create/modify symbolic link for best checkpoint
        symlink_update(
            "{}.ckpt".format(self.steps), "{}/best.ckpt".format(self.model_dir)
        )

    def init_from_checkpoint(
        self,
        path: str,
        reset_best_ckpt: bool = False,
        reset_scheduler: bool = False,
        reset_optimizer: bool = False,
    ) -> None:
        """
        Initialize the trainer from a given checkpoint file.

        This checkpoint file contains not only model parameters, but also
        scheduler and optimizer states, see `self._save_checkpoint`.

        :param path: path to checkpoint
        :param reset_best_ckpt: reset tracking of the best checkpoint,
                                use for domain adaptation with a new dev
                                set or when using a new metric for fine-tuning.
        :param reset_scheduler: reset the learning rate scheduler, and do not
                                use the one stored in the checkpoint.
        :param reset_optimizer: reset the optimizer, and do not use the one
                                stored in the checkpoint.
        """
        model_checkpoint = load_checkpoint(path=path, use_cuda=self.use_cuda)

        # restore model and optimizer parameters
        self.model.load_state_dict(model_checkpoint["model_state"])

        if not reset_optimizer:
            self.optimizer.load_state_dict(model_checkpoint["optimizer_state"])
        else:
            self.logger.info("Reset optimizer.")

        if not reset_scheduler:
            if (
                model_checkpoint["scheduler_state"] is not None
                and self.scheduler is not None
            ):
                self.scheduler.load_state_dict(model_checkpoint["scheduler_state"])
        else:
            self.logger.info("Reset scheduler.")

        # restore counts
        self.steps = model_checkpoint["steps"]
        self.total_txt_tokens = model_checkpoint["total_txt_tokens"]
        self.total_gls_tokens = model_checkpoint["total_gls_tokens"]

        if not reset_best_ckpt:
            self.best_ckpt_score = model_checkpoint["best_ckpt_score"]
            self.best_all_ckpt_scores = model_checkpoint["best_all_ckpt_scores"]
            self.best_ckpt_iteration = model_checkpoint["best_ckpt_iteration"]
        else:
            self.logger.info("Reset tracking of the best checkpoint.")

        # move parameters to cuda
        if self.use_cuda:
            self.model.cuda()


    def train_and_validate(self, train_data: Dataset, valid_data: Dataset, train_gloss2text_data: Dataset) -> None:
        """
        Train the model and validate it from time to time on the validation set.

        :param train_data: training data
        :param valid_data: validation data
        """
        train_iter = make_data_iter(
            train_data,
            batch_size=self.batch_size,
            batch_type=self.batch_type,
            train=True,
            shuffle=self.shuffle,
        )


        if train_gloss2text_data != None and self.config["training"].get("G2T_pretraining", False):

            train_g2t_iter = make_data_iter(
                train_gloss2text_data,
                batch_size=self.batch_size,
                batch_type=self.batch_type,
                train=True,
                shuffle=self.shuffle,
                sort_key="gls"
            )
            train_g2t_iter = iter(train_g2t_iter)

        epoch_no = None

        S2G_WER = 0
        for epoch_no in range(self.epochs):
            self.logger.info("EPOCH %d", epoch_no + 1)

            if self.scheduler is not None and self.scheduler_step_at == "epoch":
                self.scheduler.step(epoch=epoch_no)

            self.model.train()
            start = time.time()
            total_valid_duration = 0
            count = self.batch_multiplier - 1

            if self.do_recognition:
                processed_gls_tokens = self.total_gls_tokens
                epoch_recognition_loss = 0
            if self.do_translation:
                processed_txt_tokens = self.total_txt_tokens
                epoch_translation_loss = 0

            for batch in iter(train_iter):
                # reactivate training
                # create a Batch object from torchtext batch
                # TODO update on Gloss2Text data for easy start
                if train_gloss2text_data != None and self.config["training"].get("G2T_pretraining", False):
                    try:
                        batch_g2t = next(train_g2t_iter)
                    except StopIteration:
                        # handle the situation, e.g., reset the iterator or break the loop
                        train_g2t_iter = make_data_iter(
                            train_gloss2text_data,
                            batch_size=self.batch_size,
                            batch_type=self.batch_type,
                            train=True,
                            shuffle=self.shuffle,
                            sort_key="gls"
                        )
                        train_g2t_iter = iter(train_g2t_iter)
                        batch_g2t = next(train_g2t_iter)

                    batch_g2t = Batch_gls2text(
                        is_train=True,
                        torch_batch=batch_g2t,
                        txt_pad_index=self.txt_pad_index,
                        gla_pad_index=self.model.gls_pad_index,
                        use_cuda=self.use_cuda,
                        frame_subsampling_ratio=self.frame_subsampling_ratio,
                        random_frame_subsampling=self.random_frame_subsampling,
                        random_frame_masking_ratio=self.random_frame_masking_ratio,
                        if_MixGen=0
                    )

                    update = count == 0

                    recognition_loss_DA, translation_loss_DA = self._train_batch(
                        batch_g2t, update=update, forward_type="gloss"
                    )

                # TODO Sign2Gloss update
                batch_0 = Batch_Sign(
                    is_train=True,
                    torch_batch=batch,
                    txt_pad_index=self.txt_pad_index,
                    gla_pad_index=self.model.gls_pad_index,
                    sgn_dim=self.feature_size,
                    use_cuda=self.use_cuda,
                    frame_subsampling_ratio=self.frame_subsampling_ratio,
                    random_frame_subsampling=self.random_frame_subsampling,
                    random_frame_masking_ratio=self.random_frame_masking_ratio,
                    if_MixGen=0
                )

                update = count == 0

                recognition_loss, translation_loss = self._train_batch(
                    batch_0, update=update, forward_type = self.config["training"]["forward_type"], S2G_WER=S2G_WER
                )


                if self.do_recognition:
                    self.tb_writer.add_scalar(
                        "train/train_recognition_loss", recognition_loss, self.steps
                    )
                    epoch_recognition_loss += recognition_loss.detach().cpu().numpy()

                if self.do_translation:
                    self.tb_writer.add_scalar(
                        "train/train_translation_loss", translation_loss, self.steps
                    )
                    epoch_translation_loss += translation_loss.detach().cpu().numpy()

                count = self.batch_multiplier if update else count
                count -= 1

                if (
                    self.scheduler is not None
                    and self.scheduler_step_at == "step"
                    and update
                ):
                    self.scheduler.step()

                # log learning progress
                if self.steps % self.logging_freq == 0 and update:
                    elapsed = time.time() - start - total_valid_duration

                    log_out = "[Epoch: {:03d} Step: {:08d}] ".format(
                        epoch_no + 1, self.steps,
                    )

                    if self.do_recognition:
                        elapsed_gls_tokens = (
                            self.total_gls_tokens - processed_gls_tokens
                        )
                        processed_gls_tokens = self.total_gls_tokens
                        log_out += "Batch Recognition Loss: {:10.6f} => ".format(
                            recognition_loss
                        )
                        log_out += "Gls Tokens per Sec: {:8.0f} || ".format(
                            elapsed_gls_tokens / elapsed
                        )
                    if self.do_translation:
                        elapsed_txt_tokens = (
                            self.total_txt_tokens - processed_txt_tokens
                        )
                        processed_txt_tokens = self.total_txt_tokens
                        log_out += "Batch Translation Loss: {:10.6f} => ".format(
                            translation_loss
                        )
                        log_out += "Txt Tokens per Sec: {:8.0f} || ".format(
                            elapsed_txt_tokens / elapsed
                        )
                    log_out += "Lr: {:.6f}".format(self.optimizer.param_groups[0]["lr"])
                    self.logger.info(log_out)
                    start = time.time()
                    total_valid_duration = 0

                # validate on the entire dev set
                if self.steps % self.validation_freq == 0 and update:
                    valid_start_time = time.time()
                    # TODO (Cihan): There must be a better way of passing
                    #   these recognition only and translation only parameters!
                    #   Maybe have a NamedTuple with optional fields?
                    #   Hmm... Future Cihan's problem.
                    val_res = validate_on_data(
                        model=self.model,
                        forward_type=self.config["testing"]["forward_type"],
                        data=valid_data,
                        batch_size=self.eval_batch_size,
                        use_cuda=self.use_cuda,
                        batch_type=self.eval_batch_type,
                        dataset_version=self.dataset_version,
                        sgn_dim=self.feature_size,
                        txt_pad_index=self.txt_pad_index,
                        # Recognition Parameters
                        do_recognition=self.do_recognition,
                        recognition_loss_function=self.recognition_loss_function
                        if self.do_recognition
                        else None,
                        recognition_loss_weight=self.recognition_loss_weight
                        if self.do_recognition
                        else None,
                        recognition_beam_size=self.eval_recognition_beam_size
                        if self.do_recognition
                        else None,
                        # Translation Parameters
                        do_translation=self.do_translation,
                        translation_loss_function=self.translation_loss_function
                        if self.do_translation
                        else None,
                        translation_max_output_length=self.translation_max_output_length
                        if self.do_translation
                        else None,
                        level=self.level if self.do_translation else None,
                        translation_loss_weight=self.translation_loss_weight
                        if self.do_translation
                        else None,
                        translation_beam_size=self.eval_translation_beam_size
                        if self.do_translation
                        else None,
                        translation_beam_alpha=self.eval_translation_beam_alpha
                        if self.do_translation
                        else None,
                        frame_subsampling_ratio=self.frame_subsampling_ratio,
                    )
                    self.model.train()

                    if self.do_recognition:
                        # Log Losses and ppl
                        self.tb_writer.add_scalar(
                            "valid/valid_recognition_loss",
                            val_res["valid_recognition_loss"],
                            self.steps,
                        )
                        self.tb_writer.add_scalar(
                            "valid/wer", val_res["valid_scores"]["wer"], self.steps
                        )
                        self.tb_writer.add_scalars(
                            "valid/wer_scores",
                            val_res["valid_scores"]["wer_scores"],
                            self.steps,
                        )

                    if self.do_translation:
                        self.tb_writer.add_scalar(
                            "valid/valid_translation_loss",
                            val_res["valid_translation_loss"],
                            self.steps,
                        )
                        self.tb_writer.add_scalar(
                            "valid/valid_ppl", val_res["valid_ppl"], self.steps
                        )

                        # Log Scores
                        self.tb_writer.add_scalar(
                            "valid/chrf", val_res["valid_scores"]["chrf"], self.steps
                        )
                        self.tb_writer.add_scalar(
                            "valid/rouge", val_res["valid_scores"]["rouge"], self.steps
                        )
                        self.tb_writer.add_scalar(
                            "valid/bleu", val_res["valid_scores"]["bleu"], self.steps
                        )
                        self.tb_writer.add_scalars(
                            "valid/bleu_scores",
                            val_res["valid_scores"]["bleu_scores"],
                            self.steps,
                        )

                    if self.early_stopping_metric == "recognition_loss":
                        assert self.do_recognition
                        ckpt_score = val_res["valid_recognition_loss"]
                    elif self.early_stopping_metric == "translation_loss":
                        assert self.do_translation
                        ckpt_score = val_res["valid_translation_loss"]
                    elif self.early_stopping_metric in ["ppl", "perplexity"]:
                        assert self.do_translation
                        ckpt_score = val_res["valid_ppl"]
                    else:
                        ckpt_score = val_res["valid_scores"][self.eval_metric]

                    new_best = False
                    if self.is_best(ckpt_score):
                        self.best_ckpt_score = ckpt_score
                        self.best_all_ckpt_scores = val_res["valid_scores"]
                        self.best_ckpt_iteration = self.steps
                        self.logger.info(
                            "Hooray! New best validation result [%s]!",
                            self.early_stopping_metric,
                        )
                        if self.ckpt_queue.maxsize > 0:
                            self.logger.info("Saving new checkpoint.")
                            new_best = True
                            self._save_checkpoint()

                    if (
                        self.scheduler is not None
                        and self.scheduler_step_at == "validation"
                    ):
                        prev_lr = self.scheduler.optimizer.param_groups[0]["lr"]
                        self.scheduler.step(ckpt_score)
                        now_lr = self.scheduler.optimizer.param_groups[0]["lr"]

                        if prev_lr != now_lr:
                            if self.last_best_lr != prev_lr:
                                self.stop = False #True #@jinhui 我不想这样提前停

                    # append to validation report
                    self._add_report(
                        val_res=val_res
                        if self.do_recognition
                        else None,
                        valid_translation_loss=val_res["valid_translation_loss"]
                        if self.do_translation
                        else None,
                        valid_ppl=val_res["valid_ppl"] if self.do_translation else None,
                        eval_metric=self.eval_metric,
                        new_best=new_best,
                    )
                    valid_duration = time.time() - valid_start_time
                    total_valid_duration += valid_duration
                    self.logger.info(
                        "Validation result at epoch %3d, step %8d: duration: %.4fs\n\t"
                        "Recognition Beam Size: %d\t"
                        "Translation Beam Size: %d\t"
                        "Translation Beam Alpha: %d\n\t"
                        "Recognition Loss: %4.5f\t"
                        "Translation Loss: %4.5f\t"
                        "PPL: %4.5f\n\t"
                        "Eval Metric: %s\n\t"
                        "WER %3.2f\t(DEL: %3.2f,\tINS: %3.2f,\tSUB: %3.2f)\n\t"
                        "BLEU-4 %.2f\t(BLEU-1: %.2f,\tBLEU-2: %.2f,\tBLEU-3: %.2f,\tBLEU-4: %.2f)\n\t"
                        "CHRF %.2f\t"
                        "ROUGE %.2f",
                        epoch_no + 1,
                        self.steps,
                        valid_duration,
                        self.eval_recognition_beam_size if self.do_recognition else -1,
                        self.eval_translation_beam_size if self.do_translation else -1,
                        self.eval_translation_beam_alpha if self.do_translation else -1,
                        val_res["valid_recognition_loss"]
                        if self.do_recognition
                        else -1,
                        val_res["valid_translation_loss"]
                        if self.do_translation
                        else -1,
                        val_res["valid_ppl"] if self.do_translation else -1,
                        self.eval_metric.upper(),
                        # WER
                        val_res["valid_scores"]["wer"] if self.do_recognition else -1,
                        val_res["valid_scores"]["wer_scores"]["del_rate"]
                        if self.do_recognition
                        else -1,
                        val_res["valid_scores"]["wer_scores"]["ins_rate"]
                        if self.do_recognition
                        else -1,
                        val_res["valid_scores"]["wer_scores"]["sub_rate"]
                        if self.do_recognition
                        else -1,
                        # BLEU
                        val_res["valid_scores"]["bleu"] if self.do_translation else -1,
                        val_res["valid_scores"]["bleu_scores"]["bleu1"]
                        if self.do_translation
                        else -1,
                        val_res["valid_scores"]["bleu_scores"]["bleu2"]
                        if self.do_translation
                        else -1,
                        val_res["valid_scores"]["bleu_scores"]["bleu3"]
                        if self.do_translation
                        else -1,
                        val_res["valid_scores"]["bleu_scores"]["bleu4"]
                        if self.do_translation
                        else -1,
                        # Other
                        val_res["valid_scores"]["chrf"] if self.do_translation else -1,
                        val_res["valid_scores"]["rouge"] if self.do_translation else -1,
                    )

                    self._log_examples(
                        sequences=[s for s in valid_data.sequence],
                        gls_references=val_res["gls_ref"]
                        if self.do_recognition
                        else None,
                        gls_hypotheses=val_res["gls_hyp"]
                        if self.do_recognition
                        else None,
                        txt_references=val_res["txt_ref"]
                        if self.do_translation
                        else None,
                        txt_hypotheses=val_res["txt_hyp"]
                        if self.do_translation
                        else None,
                    )

                    valid_seq = [s for s in valid_data.sequence]

                    S2G_WER = val_res["valid_scores"]["wer"]
                    if self.do_recognition:
                        self._store_outputs(
                            "dev.hyp.gls", valid_seq, val_res["gls_hyp"], "gls"
                        )
                        self._store_outputs(
                            "references.dev.gls", valid_seq, val_res["gls_ref"]
                        )

                    if self.do_translation:
                        self._store_outputs(
                            "dev.hyp.txt", valid_seq, val_res["txt_hyp"], "txt"
                        )
                        self._store_outputs(
                            "references.dev.txt", valid_seq, val_res["txt_ref"]
                        )

                if self.stop:
                    break
            if self.stop:
                if (
                    self.scheduler is not None
                    and self.scheduler_step_at == "validation"
                    and self.last_best_lr != prev_lr
                ):
                    self.logger.info(
                        "Training ended since there were no improvements in"
                        "the last learning rate step: %f",
                        prev_lr,
                    )
                else:
                    self.logger.info(
                        "Training ended since minimum lr %f was reached.",
                        self.learning_rate_min,
                    )
                break

            self.logger.info(
                "Epoch %3d: Total Training Recognition Loss %.2f "
                " Total Training Translation Loss %.2f ",
                epoch_no + 1,
                epoch_recognition_loss if self.do_recognition else -1,
                epoch_translation_loss if self.do_translation else -1,
            )
        else:
            self.logger.info("Training ended after %3d epochs.", epoch_no + 1)
        self.logger.info(
            "Best validation result at step %8d: %6.2f %s.",
            self.best_ckpt_iteration,
            self.best_ckpt_score,
            self.early_stopping_metric,
        )

        self.tb_writer.close()  # close Tensorboard writer

    def get_loss_for_mixup(
        self,
        batch: Batch,
        recognition_loss_function: nn.Module,
        translation_loss_function: nn.Module,
        recognition_loss_weight: float,
        translation_loss_weight: float,
        forward_type=None,
    ) -> (Tensor, Tensor):
        """
        Compute non-normalized loss and number of tokens for a batch

        :param batch: batch to compute loss for
        :param recognition_loss_function: Sign Language Recognition Loss Function (CTC)
        :param translation_loss_function: Sign Language Translation Loss Function (XEntropy)
        :param recognition_loss_weight: Weight for recognition loss
        :param translation_loss_weight: Weight for translation loss
        :return: recognition_loss: sum of losses over sequences in the batch
        :return: translation_loss: sum of losses over non-pad elements in the batch
        """

        #TODO get sign forward
        decoder_outputs_sgnBase, gloss_probabilities_sgnBase = self.model.forward(
            sgn=batch.sgn,
            sgn_mask=batch.sgn_mask,
            sgn_lengths=batch.sgn_lengths,
            txt_input=batch.txt_input,
            txt_mask=batch.txt_mask,
        )

        if self.do_recognition:
            assert gloss_probabilities_sgnBase is not None
            # Calculate Recognition Loss
            recognition_loss_sgnBase = (
                    recognition_loss_function(
                        gloss_probabilities_sgnBase,
                        batch.gls,
                        batch.sgn_lengths.long(),
                        batch.gls_lengths.long(),
                    )
                    * recognition_loss_weight
            )
        else:
            recognition_loss_sgnBase = None

        if self.do_translation:
            assert decoder_outputs_sgnBase is not None
            word_outputs_sgnBase, _, _, _ = decoder_outputs_sgnBase
            # Calculate Translation Loss
            txt_log_probs_sgnBase = F.log_softmax(word_outputs_sgnBase, dim=-1)
            translation_loss_sgnBase = (
                    translation_loss_function(txt_log_probs_sgnBase, batch.txt)
                    * translation_loss_weight
            )
        else:
            translation_loss_sgnBase = None

        # @jinhui waiting 考虑每个 batch 中是否做mixup
        if recognition_loss_sgnBase < 5 or random.randint(0,10) < 5:
            aligner = "path_wise"
            # T x B x C
            gloss_probabilities = gloss_probabilities_sgnBase
            # Turn it into B x T x C
            gloss_probabilities = gloss_probabilities.permute(1, 0, 2)

            if aligner == "frames_wise":
                # T x N
                gloss_predict = torch.argmax(gloss_probabilities, dim=-1)  # 可以是可以，但是没有CTC
            elif aligner == "path_wise":
                alignment, gloss_predict = self.ctc_alignment(log_probs=gloss_probabilities,
                                          input_lengths=batch.sgn_lengths.long(),
                                          targets=batch.gls,
                                          target_lengths=batch.gls_lengths.long())
                pass

            #　TODO get mixup embedding
            glosses_embedding = self.model.gloss_embed(x=gloss_predict, mask=batch.sgn_mask)
            sign_embedding = self.model.sgn_embed(x=batch.sgn, mask=batch.sgn_mask)
            # @https://blog.csdn.net/weixin_44575152/article/details/123880800
            mixup_ratio = self.config["training"]["mixup_ratio"]
            if mixup_ratio == "auto":
                mixup_ratio = 0
            elif mixup_ratio == "beta ":
                mixup_ratio = np.random.beta(0.5, 0.5)
            else:
                pass
            mix_mask_sgn = gloss_predict.ge(1 - mixup_ratio)
            mix_mask_gloss = ~mix_mask_sgn
            cc = torch.stack((mix_mask_sgn, mix_mask_gloss), dim=1).permute(0, 2, 1) # N, T, 2
            xx = torch.stack((sign_embedding, glosses_embedding), dim=2)
            bb = xx.permute(3, 0, 1, 2)
            shape_x = bb.shape
            mix_sign_embedding = torch.masked_select(bb, cc).reshape(shape_x[0:-1]).permute(1, 2, 0)

            # TODO Forward
            encoder_output, encoder_hidden = self.model.encode(
                sgn=mix_sign_embedding, sgn_mask=batch.sgn_mask, sgn_length=batch.sgn_lengths
            )

            if self.do_translation:
                unroll_steps = batch.txt_input.size(1)
                decoder_outputs_mixBase = self.model.decode(
                    encoder_output=encoder_output,
                    encoder_hidden=encoder_hidden,
                    sgn_mask=batch.sgn_mask,
                    txt_input=batch.txt_input,
                    unroll_steps=unroll_steps,
                    txt_mask=batch.txt_mask,
                )
            else:
                decoder_outputs_mixBase = 0

            if self.do_translation:
                assert decoder_outputs_mixBase is not None
                word_outputs_mixBase, _, _, _ = decoder_outputs_mixBase
                # Calculate Translation Loss
                txt_log_probs_mixBase = F.log_softmax(word_outputs_mixBase, dim=-1)
                translation_loss_mixBase = (
                        translation_loss_function(txt_log_probs_mixBase, batch.txt)
                        * translation_loss_weight
                )

                # 计算 JS loss

                txt_probs_1 = torch.exp(txt_log_probs_sgnBase)
                txt_probs_2 = torch.exp(txt_log_probs_mixBase)
                # 计算混合分布
                probs_mixture = torch.div(torch.add(txt_probs_1, txt_probs_2), 2)
                txt_log_probs_mixture = torch.log(probs_mixture + 1e-8)

                # 计算 KL divergence
                kl_div1 = torch.sum(torch.sum(txt_probs_1 * (txt_log_probs_sgnBase - txt_log_probs_mixture), dim=-1),
                                    dim=-1)
                kl_div2 = torch.sum(torch.sum(txt_probs_2 * (txt_log_probs_mixBase - txt_log_probs_mixture), dim=-1),
                                    dim=-1)

                # 计算 JS divergence
                js_loss = torch.sum(torch.div(torch.add(kl_div1, kl_div2), 2), dim=-1) * translation_loss_weight * 15

                translation_loss_mixBase = js_loss + translation_loss_mixBase


                # 计算 JS loss
                # js_loss = torch.div(torch.add(kl_div1, kl_div2), 2) * translation_loss_weight #/ torch.sqrt(txt_log_probs_mixBase.shape())
                gloss_emb_probs_1 = torch.exp(mix_sign_embedding)
                sign_emb_probs_2 = torch.exp(sign_embedding)
                # 计算混合分布
                emb_mixture = torch.div(torch.add(gloss_emb_probs_1, sign_emb_probs_2), 2)
                emb_log_probs_mixture = torch.log(emb_mixture + 1e-8)

                # 计算 KL divergence
                emb_kl_div1 = torch.sum(torch.sum(gloss_emb_probs_1 * (mix_sign_embedding - emb_log_probs_mixture), dim=-1),
                                    dim=-1)
                emb_kl_div2 = torch.sum(torch.sum(sign_emb_probs_2 * (sign_embedding - emb_log_probs_mixture), dim=-1),
                                    dim=-1)

                # 计算 JS divergence
                emb_js_loss = torch.sum(torch.div(torch.add(emb_kl_div1, emb_kl_div2), 2), dim=-1) * translation_loss_weight * 15

                translation_loss_mixBase = translation_loss_mixBase + emb_js_loss
            else:
                translation_loss_mixBase = 0

        else:
            translation_loss_mixBase = 0

        return  recognition_loss_sgnBase, translation_loss_sgnBase + translation_loss_mixBase

    def ctc_alignment(self, log_probs, input_lengths, targets, target_lengths):
        batch_alignment = []
        batch_label_at_pos = []

        max_input_length = max(input_lengths)

        # Iterate through each batch
        for log_prob, length, target, target_length in zip(log_probs, input_lengths, targets, target_lengths):
            log_prob = log_prob[:length].cpu()
            gloss_padding_index = target[-1]
            target = target[:target_length].tolist()

            input_length = len(log_prob)
            target_length = len(target)

            alignment = []
            label_at_pos = []

            positions = np.linspace(0, input_length, target_length + 1).astype(int)

            for i in range(target_length):
                start = positions[i]
                end = positions[i + 1]
                label = target[i]

                if i < target_length - 1:
                    next_label = target[i + 1]

                    if start < end:
                        best_idx = np.argmax(log_prob[start:end, label].detach().numpy()) + start
                    else:
                        best_idx = start

                    search_range_end = min(end + (input_length - end) // 2, input_length)
                    if end < search_range_end:
                        next_best_idx = np.argmax(log_prob[end:search_range_end, next_label].detach().numpy()) + end
                    else:
                        next_best_idx = end

                    if best_idx < next_best_idx:
                        split_point = np.argmax(
                            log_prob[best_idx:next_best_idx, label].detach().numpy() - log_prob[best_idx:next_best_idx,
                                                                                       next_label].detach().numpy()) + best_idx
                    else:
                        split_point = best_idx + 1
                else:
                    split_point = input_length

                if split_point == start:
                    split_point += 1

                alignment.extend([i] * (split_point - len(alignment)))
                label_at_pos.extend([label] * (split_point - len(label_at_pos)))

            # Padding the alignment and label_at_pos sequences to the same length
            alignment.extend([-1] * (max_input_length - len(alignment)))
            label_at_pos.extend([gloss_padding_index] * (max_input_length - len(label_at_pos)))

            batch_alignment.append(alignment)
            batch_label_at_pos.append(label_at_pos)

        return torch.tensor(batch_alignment).cuda(), torch.tensor(batch_label_at_pos).cuda()

    def get_alignment(self, log_prob, target, input_length, target_length):
        alignment = []
        label_at_pos = []

        for i in range(target_length):
            start = i * (input_length // target_length)
            end = (i + 1) * (input_length // target_length) if i < target_length - 1 else input_length

            if i == target_length - 1:
                alignment.extend([i] * (input_length - len(alignment)))
                label_at_pos.extend([target[i].item()] * (input_length - len(label_at_pos)))
            else:
                next_best_idx = np.argmax(log_prob[end:, target[i + 1]].detach().cpu().numpy()) + end
                split_point = np.argmax(
                    log_prob[start:next_best_idx, target[i]] - log_prob[start:next_best_idx, target[i + 1]]) + start
                alignment.extend([i] * (split_point - len(alignment) + 1))
                label_at_pos.extend([target[i].item()] * (split_point - len(label_at_pos) + 1))

        return alignment, label_at_pos

    def _train_batch(self, batch: Batch, update: bool = True, forward_type="sign", S2G_WER=0) -> (Tensor, Tensor):
        """
        Train the model on one batch: Compute the loss, make a gradient step.

        :param batch: training batch
        :param update: if False, only store gradient. if True also make update
        :return normalized_recognition_loss: Normalized recognition loss
        :return normalized_translation_loss: Normalized translation loss
        """

        if forward_type == "gloss" or forward_type == "sign":
            forward_function = self.model.get_loss_for_batch
        elif forward_type == "mixup":
            if S2G_WER < 40 or random.randint(0,10) < 3:
                forward_function = self.get_loss_for_mixup
            else:
                forward_type = "sign"
                forward_function = self.model.get_loss_for_batch

        else:
            raise NotImplementedError("Not {} forward process".format(forward_type))

        recognition_loss, translation_loss = forward_function(
            batch=batch,
            forward_type=forward_type,
            recognition_loss_function=self.recognition_loss_function
            if self.do_recognition
            else None,
            translation_loss_function=self.translation_loss_function
            if self.do_translation
            else None,
            recognition_loss_weight=self.recognition_loss_weight
            if self.do_recognition
            else None,
            translation_loss_weight=self.translation_loss_weight
            if self.do_translation
            else None,
        )
        # normalize translation loss
        if self.do_translation:
            if self.translation_normalization_mode == "batch":
                txt_normalization_factor = batch.num_seqs
            elif self.translation_normalization_mode == "tokens":
                txt_normalization_factor = batch.num_txt_tokens
            else:
                raise NotImplementedError("Only normalize by 'batch' or 'tokens'")

            # division needed since loss.backward sums the gradients until updated
            normalized_translation_loss = translation_loss / (
                txt_normalization_factor * self.batch_multiplier
            )
        else:
            normalized_translation_loss = 0

        # TODO (Cihan): Add Gloss Token normalization (?)
        #   I think they are already being normalized by batch
        #   I need to think about if I want to normalize them by token.
        if self.do_recognition:
            normalized_recognition_loss = recognition_loss / self.batch_multiplier
        else:
            normalized_recognition_loss = 0

        total_loss = normalized_recognition_loss + normalized_translation_loss
        # compute gradients
        total_loss.backward()

        if self.clip_grad_fun is not None:
            # clip gradients (in-place)
            self.clip_grad_fun(params=self.model.parameters())

        if update:
            # make gradient step
            self.optimizer.step()
            self.optimizer.zero_grad()

            # increment step counter
            self.steps += 1

        # increment token counter
        if self.do_recognition:
            self.total_gls_tokens += batch.num_gls_tokens
        if self.do_translation:
            self.total_txt_tokens += batch.num_txt_tokens

        return normalized_recognition_loss, normalized_translation_loss

    def _add_report(
        self,
        val_res: Dict,
        valid_translation_loss: float,
        valid_ppl: float,
        eval_metric: str,
        new_best: bool = False,
    ) -> None:
        """
        Append a one-line report to validation logging file.

        :param valid_scores: Dictionary of validation scores
        :param valid_recognition_loss: validation loss (sum over whole validation set)
        :param valid_translation_loss: validation loss (sum over whole validation set)
        :param valid_ppl: validation perplexity
        :param eval_metric: evaluation metric, e.g. "bleu"
        :param new_best: whether this is a new best model
        """
        valid_scores = val_res["valid_scores"]
        valid_recognition_loss = val_res["valid_recognition_loss"]
        current_lr = -1
        # ignores other param groups for now
        for param_group in self.optimizer.param_groups:
            current_lr = param_group["lr"]

        if new_best:
            self.last_best_lr = current_lr

        if current_lr < self.learning_rate_min:
            self.stop = True
        s2t_acc = metrics.bleu(val_res["gls_hyp"], val_res["gls_ref"])["bleu4"]
        with open(self.valid_report_file, "a", encoding="utf-8") as opened_file:
            opened_file.write(
                "Steps: {}\t"
                "Recognition BLEU-2 {:.2f}\t"
                "Recognition Loss: {:.5f}\t"
                "Translation Loss: {:.5f}\t"
                "PPL: {:.5f}\t"
                "Eval Metric: {}\t"
                "WER {:.2f}\t(DEL: {:.2f},\tINS: {:.2f},\tSUB: {:.2f})\t"
                "BLEU-4 {:.2f}\t(BLEU-1: {:.2f},\tBLEU-2: {:.2f},\tBLEU-3: {:.2f},\tBLEU-4: {:.2f})\t"
                "CHRF {:.2f}\t"
                "ROUGE {:.2f}\t"
                "LR: {:.8f}\t{}\n".format(
                    self.steps,
                    s2t_acc if self.do_recognition else -1,
                    valid_recognition_loss if self.do_recognition else -1,
                    valid_translation_loss if self.do_translation else -1,
                    valid_ppl if self.do_translation else -1,
                    eval_metric,
                    # WER
                    valid_scores["wer"] if self.do_recognition else -1,
                    valid_scores["wer_scores"]["del_rate"]
                    if self.do_recognition
                    else -1,
                    valid_scores["wer_scores"]["ins_rate"]
                    if self.do_recognition
                    else -1,
                    valid_scores["wer_scores"]["sub_rate"]
                    if self.do_recognition
                    else -1,
                    # BLEU
                    valid_scores["bleu"] if self.do_translation else -1,
                    valid_scores["bleu_scores"]["bleu1"] if self.do_translation else -1,
                    valid_scores["bleu_scores"]["bleu2"] if self.do_translation else -1,
                    valid_scores["bleu_scores"]["bleu3"] if self.do_translation else -1,
                    valid_scores["bleu_scores"]["bleu4"] if self.do_translation else -1,
                    # Other
                    valid_scores["chrf"] if self.do_translation else -1,
                    valid_scores["rouge"] if self.do_translation else -1,
                    current_lr,
                    "*" if new_best else "",
                )
            )

    def _log_parameters_list(self) -> None:
        """
        Write all model parameters (name, shape) to the log.
        """
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info("Total params: %d", n_params)
        trainable_params = [
            n for (n, p) in self.model.named_parameters() if p.requires_grad
        ]
        self.logger.info("Trainable parameters: %s", sorted(trainable_params))
        assert trainable_params

    def _log_examples(
        self,
        sequences: List[str],
        gls_references: List[str],
        gls_hypotheses: List[str],
        txt_references: List[str],
        txt_hypotheses: List[str],
    ) -> None:
        """
        Log `self.num_valid_log` number of samples from valid.

        :param sequences: sign video sequence names (list of strings)
        :param txt_hypotheses: decoded txt hypotheses (list of strings)
        :param txt_references: decoded txt references (list of strings)
        :param gls_hypotheses: decoded gls hypotheses (list of strings)
        :param gls_references: decoded gls references (list of strings)
        """

        if self.do_recognition:
            assert len(gls_references) == len(gls_hypotheses)
            num_sequences = len(gls_hypotheses)
        if self.do_translation:
            assert len(txt_references) == len(txt_hypotheses)
            num_sequences = len(txt_hypotheses)

        rand_idx = np.sort(np.random.permutation(num_sequences)[: self.num_valid_log])
        self.logger.info("Logging Recognition and Translation Outputs")
        self.logger.info("=" * 120)
        for ri in rand_idx:
            self.logger.info("Logging Sequence: %s", sequences[ri])
            if self.do_recognition:
                gls_res = wer_single(r=gls_references[ri], h=gls_hypotheses[ri])
                self.logger.info(
                    "\tGloss Reference :\t%s", gls_res["alignment_out"]["align_ref"]
                )
                self.logger.info(
                    "\tGloss Hypothesis:\t%s", gls_res["alignment_out"]["align_hyp"]
                )
                self.logger.info(
                    "\tGloss Alignment :\t%s", gls_res["alignment_out"]["alignment"]
                )
            if self.do_recognition and self.do_translation:
                self.logger.info("\t" + "-" * 116)
            if self.do_translation:
                txt_res = wer_single(r=txt_references[ri], h=txt_hypotheses[ri])
                self.logger.info(
                    "\tText Reference  :\t%s", txt_res["alignment_out"]["align_ref"]
                )
                self.logger.info(
                    "\tText Hypothesis :\t%s", txt_res["alignment_out"]["align_hyp"]
                )
                self.logger.info(
                    "\tText Alignment  :\t%s", txt_res["alignment_out"]["alignment"]
                )
            self.logger.info("=" * 120)

    def _store_outputs(
        self, tag: str, sequence_ids: List[str], hypotheses: List[str], sub_folder=None
    ) -> None:
        """
        Write current validation outputs to file in `self.model_dir.`

        :param hypotheses: list of strings
        """
        if sub_folder:
            out_folder = os.path.join(self.model_dir, sub_folder)
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            current_valid_output_file = "{}/{}.{}".format(out_folder, self.steps, tag)
        else:
            out_folder = self.model_dir
            current_valid_output_file = "{}/{}".format(out_folder, tag)

        with open(current_valid_output_file, "w", encoding="utf-8") as opened_file:
            for seq, hyp in zip(sequence_ids, hypotheses):
                opened_file.write("{}|{}\n".format(seq, hyp))

    def training_on_gls2text(self, model=None, trainset=None):
        if trainset != None:
            train_g2t_iter = make_data_iter(
                trainset,
                batch_size=self.batch_size,
                batch_type=self.batch_type,
                train=True,
                shuffle=self.shuffle,
                sort_key="gls"
            )


        for batch in iter(train_g2t_iter):
            # reactivate training
            # create a Batch object from torchtext batch
            batch_0 = Batch_gls2text(
                is_train=True,
                torch_batch=batch,
                txt_pad_index=self.txt_pad_index,
                gla_pad_index=self.model.gls_pad_index,
                use_cuda=self.use_cuda,
                frame_subsampling_ratio=self.frame_subsampling_ratio,
                random_frame_subsampling=self.random_frame_subsampling,
                random_frame_masking_ratio=self.random_frame_masking_ratio,
                if_MixGen=0
            )

            update = 1

            recognition_loss, translation_loss = self._train_batch(
                batch_0, update=update, forward_type="gloss"
            )

        pass

def train(cfg_file: str) -> None:
    """
    Main training function. After training, also test on test data if given.

    :param cfg_file: path to configuration yaml file
    """
    cfg = load_config(cfg_file)

    # set the random seed
    set_seed(seed=cfg["training"].get("random_seed", 42))

    train_data, dev_data, test_data, gls_vocab, txt_vocab, train_gloss2text_data = load_data(
        data_cfg=cfg["data"]
    )

    # build model and load parameters into it
    do_recognition = cfg["training"].get("recognition_loss_weight", 1.0) > 0.0
    do_translation = cfg["training"].get("translation_loss_weight", 1.0) > 0.0
    build_type = cfg["model"].get("build_type", "build_model")
    if build_type :
        model = build_model_mixup(
            cfg=cfg["model"],
            gls_vocab=gls_vocab,
            txt_vocab=txt_vocab,
            sgn_dim=sum(cfg["data"]["feature_size"])
            if isinstance(cfg["data"]["feature_size"], list)
            else cfg["data"]["feature_size"],
            do_recognition=do_recognition,
            do_translation=do_translation,
        )
    else:
        model = build_model(
            cfg=cfg["model"],
            gls_vocab=gls_vocab,
            txt_vocab=txt_vocab,
            sgn_dim=sum(cfg["data"]["feature_size"])
            if isinstance(cfg["data"]["feature_size"], list)
            else cfg["data"]["feature_size"],
            do_recognition=do_recognition,
            do_translation=do_translation,
        )

    # for training management, e.g. early stopping and model selection
    trainer = TrainManager(model=model, config=cfg)

    # store copy of original training config in model dir
    shutil.copy2(cfg_file, trainer.model_dir + "/config.yaml")

    # log all entries of config
    log_cfg(cfg, trainer.logger)

    log_data_info(
        train_data=train_data,
        valid_data=dev_data,
        test_data=test_data,
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
        logging_function=trainer.logger.info,
    )

    trainer.logger.info(str(model))

    # store the vocabs
    gls_vocab_file = "{}/gls.vocab".format(cfg["training"]["model_dir"])
    gls_vocab.to_file(gls_vocab_file)
    txt_vocab_file = "{}/txt.vocab".format(cfg["training"]["model_dir"])
    txt_vocab.to_file(txt_vocab_file)

    # train the model
    trainer.train_and_validate(train_data=train_data, valid_data=dev_data, train_gloss2text_data=train_gloss2text_data)
    # Delete to speed things up as we don't need training data anymore
    del train_data, dev_data, test_data, train_gloss2text_data

    # predict with the best model on validation and test
    # (if test data is available)
    ckpt = "{}/{}.ckpt".format(trainer.model_dir, trainer.best_ckpt_iteration)
    output_name = "best.IT_{:08d}".format(trainer.best_ckpt_iteration)
    output_path = os.path.join(trainer.model_dir, output_name)
    logger = trainer.logger
    del trainer
    # print(ckpt)
    test(cfg_file, ckpt=ckpt, output_path=output_path, logger=logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Joey-NMT")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        type=str,
        help="Training configuration file (yaml).",
    )
    parser.add_argument(
        "--gpu_id", type=str, default="0", help="gpu to run your job on"
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    train(cfg_file=args.config)
