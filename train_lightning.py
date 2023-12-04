import os
from typing import Any
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch
from metric.FRC import compute_FRC, compute_FRC_mp
from metric.FRD import compute_FRD, compute_FRD_mp
from metric.FRDvs import compute_FRDvs
from metric.FRVar import compute_FRVar
from metric.S_MSE import compute_s_mse
from metric.TLCC import compute_TLCC, compute_TLCC_mp
from model.losses import VAELoss, div_loss
import wandb
from dataset import ReactDataModule
from render import Render


class ModelLightning(pl.LightningModule):
    def __init__(
        self, model, criterion, learning_rate=1e-3, div_p=0.1, weight_decay=0.0,
        binarize=False, render=None, out_dir=None, len_val_ds=0, test_extend_factor=1
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.div_p = div_p
        self.weight_decay = weight_decay
        
        self.binarize = binarize
        self.render = render
        self.out_dir = out_dir
        self.len_val_ds = len_val_ds
        
        os.makedirs(self.out_dir, exist_ok=True)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        (
            speaker_video_clip,
            speaker_audio_clip,
            _,
            _,
            _,
            _,
            listener_emotion,
            listener_3dmm,
            _,
        ) = batch

        (
            loss,
            rec_loss,
            kld_loss,
            listener_3dmm_out,
            listener_emotion_out,
            listener_3dmm,
            listener_emotion,
        ) = self._common_step(batch, batch_idx)

        with torch.no_grad():
            listener_3dmm_out_, listener_emotion_out_, _ = self.model(
                speaker_video_clip, speaker_audio_clip
            )

        d_loss = div_loss(listener_3dmm_out_, listener_3dmm_out) + div_loss(
            listener_emotion_out_, listener_emotion_out
        )

        loss = loss + self.div_p * d_loss

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_rec_loss",
            rec_loss,
            on_step=True,
            on_epoch=True,
            # prog_bar=True,
            logger=True,
        )
        self.log(
            "train_kld_loss",
            kld_loss,
            on_step=True,
            on_epoch=True,
            # prog_bar=True,
            logger=True,
        )
        self.log(
            "train_div_loss",
            d_loss,
            on_step=True,
            on_epoch=True,
            # prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        (
            loss,
            rec_loss,
            kld_loss,
            listener_3dmm_out,
            listener_emotion_out,
            listener_3dmm,
            listener_emotion,
        ) = self._common_step(batch, batch_idx)

        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True
        )
        self.log(
            "val_rec_loss",
            rec_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "val_kld_loss",
            kld_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        return loss
    
    def on_test_start(self) -> None:
        self.listener_emotion_gt_list = []
        self.listener_emotion_pred_list = []
        self.speaker_emotion_list = []
        self.all_listener_emotion_pred_list = []
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):

        (speaker_video_clip,
        speaker_audio_clip,
        speaker_emotion,
        _,
        listener_video_clip,
        _,
        listener_emotion,
        listener_3dmm,
        listener_references,) = batch
        
        
        prediction = self.model(
                speaker_video=speaker_video_clip,
                speaker_audio=speaker_audio_clip,
                speaker_emotion=speaker_emotion,
                listener_emotion=listener_emotion,
            )
        
        listener_3dmm_out, listener_emotion_out, distribution = prediction
        loss, rec_loss, kld_loss = self.criterion(
                    listener_emotion,
                    listener_3dmm,
                    listener_emotion_out,
                    listener_3dmm_out,
                    distribution,
                )
        
        # self.log(
        #     "val_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True
        # )
        # self.log(
        #     "val_rec_loss",
        #     rec_loss,
        #     on_step=True,
        #     on_epoch=True,
        #     prog_bar=False,
        #     logger=True,
        # )
        # self.log(
        #     "val_kld_loss",
        #     kld_loss,
        #     on_step=True,
        #     on_epoch=True,
        #     prog_bar=False,
        #     logger=True,
        # )
        
        
        if self.binarize:
            listener_emotion_out[:, :, :15] = torch.round(
                listener_emotion_out[:, :, :15]
            )
        B = speaker_video_clip.shape[0]
        if (batch_idx % 25) == 0:
            for bs in range(B):
                self.render.rendering_for_fid(
                    self.out_dir,
                    "{}_b{}_ind{}".format(
                        "val", str(batch_idx + 1), str(bs + 1)
                    ),
                    listener_3dmm_out[bs],
                    speaker_video_clip[bs],
                    listener_references[bs],
                    listener_video_clip[bs, :750],
                )
        self.listener_emotion_pred_list.append(listener_emotion_out.cpu())

        if dataloader_idx < 1:
            self.listener_emotion_gt_list.append(listener_emotion.cpu())
            self.speaker_emotion_list.append(speaker_emotion.cpu())
    
    def on_test_end(self) -> None:
        listener_emotion_pred = torch.cat(self.listener_emotion_pred_list, dim=0)
        listener_emotion_gt = torch.cat(self.listener_emotion_gt_list, dim=0)
        speaker_emotion_gt = torch.cat(self.speaker_emotion_list, dim=0)
        self.all_listener_emotion_pred_list.append(listener_emotion_pred.unsqueeze(1))

        all_listener_emotion_pred = torch.cat(self.all_listener_emotion_pred_list, dim=1)
        
        print("listener_emotion_pred.shape", listener_emotion_pred.shape)
        print("listener_emotion_gt.shape", listener_emotion_gt.shape)
        print("speaker_emotion_gt.shape", speaker_emotion_gt.shape)
        print("all_listener_emotion_pred.shape", all_listener_emotion_pred.shape)
        
        p = args.num_workers
        
        N = speaker_emotion_gt.shape[0]
        clip_length = speaker_emotion_gt.shape[1]
        emotion_dim = speaker_emotion_gt.shape[2]
        
        reshaped_all_listener_emotion_pred = all_listener_emotion_pred.reshape(N, -1, clip_length, emotion_dim)

        # If you have problems running function compute_TLCC_mp, please replace this function with function compute_TLCC
        TLCC = compute_TLCC_mp(all_listener_emotion_pred, speaker_emotion_gt, 
                            p=p
                            )

        # If you have problems running function compute_FRC_mp, please replace this function with function compute_FRC
        FRC = compute_FRC_mp(
            args, all_listener_emotion_pred, listener_emotion_gt, val_test="val", 
            p=p
        )

        # If you have problems running function compute_FRD_mp, please replace this function with function compute_FRD
        FRD = compute_FRD_mp(
            args, all_listener_emotion_pred, listener_emotion_gt, val_test="val", 
            p=p
        )

        FRDvs = compute_FRDvs(all_listener_emotion_pred)
        FRVar = compute_FRVar(all_listener_emotion_pred)
        smse = compute_s_mse(all_listener_emotion_pred)
        
        
        self.logger.log_metrics({
            "TLCC": TLCC, 
            "FRC": FRC, 
            "FRD": FRD, 
            "FRDvs": FRDvs, 
            "FRVar": FRVar, 
            "s_mse": smse})
        

    def _common_step(self, batch, batch_idx):
        (
            speaker_video_clip,
            speaker_audio_clip,
            _,
            _,
            _,
            _,
            listener_emotion,
            listener_3dmm,
            _,
        ) = batch
        listener_3dmm_out, listener_emotion_out, distribution = self.model(
            speaker_video_clip, speaker_audio_clip
        )
        loss, rec_loss, kld_loss = self.criterion(
            listener_emotion,
            listener_3dmm,
            listener_emotion_out,
            listener_3dmm_out,
            distribution,
        )

        return (
            loss,
            rec_loss,
            kld_loss,
            listener_3dmm_out,
            listener_emotion_out,
            listener_3dmm,
            listener_emotion,
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )


if __name__ == "__main__":
    from train import parse_arg

    from config import ACCELERATOR, DEVICES, PRECISION, BATCH_SIZE

    from pytorch_lightning.strategies import DeepSpeedStrategy
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

    torch.set_float32_matmul_precision("medium")

    # wandb.init(project="react-baseline",)

    # logger = WandbLogger(project="react-baseline")
    logger = TensorBoardLogger("tb_logs", name="react-baseline_overfitonebatch")
    checkpoint_callback = ModelCheckpoint(
        dirpath="/home/tien/playground_facereconstruction/checkpoint/react_baseline_transformerVAE",
        monitor="val_loss",
    )

    args = parse_arg()
    args.dataset_path = "/home/tien/playground_facereconstruction/data/react_2024"
    args.batch_size = BATCH_SIZE
    args.out_dir = "/home/tien/playground_facereconstruction/output/react_baseline_transformerVAE/test"
    args.test_extend_factor = 10

    from model import TransformerVAE

    pmt_model = TransformerVAE(
        img_size=args.img_size,
        audio_dim=args.audio_dim,
        output_3dmm_dim=args._3dmm_dim,
        output_emotion_dim=args.emotion_dim,
        feature_dim=args.feature_dim,
        seq_len=args.max_seq_len,
        online=args.online,
        window_size=args.window_size,
        device=args.device,
    )
    


    criterion = VAELoss(args.kl_p)
    
    render = Render("cuda")
    

    datamodule = ReactDataModule(conf=args)

    model = ModelLightning(
        pmt_model,
        criterion,
        learning_rate=args.learning_rate,
        div_p=args.div_p,
        weight_decay=args.weight_decay,
        render=render,
        out_dir=args.out_dir,
    )


    trainer = pl.Trainer(
        accelerator=ACCELERATOR,
        # strategy='auto',
        devices=DEVICES,
        min_epochs=1,
        max_epochs=100,
        # precision=PRECISION,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5), checkpoint_callback],
        logger=logger,
        check_val_every_n_epoch=5,
        log_every_n_steps=5,
        # fast_dev_run=True,
        # overfit_batches=1,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
