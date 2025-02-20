from pyannote.database import get_protocol, FileFinder, registry
from pyannote.audio.tasks import (
    PixIT,
)

from pytorch_lightning.loggers import TensorBoardLogger
from pyannote.audio.utils.parameters import Parameters
from argparse import ArgumentParser
import pytorch_lightning as pl

from pyannote.audio.models.separation import ToTaToNet2,ToTaToNet

from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
import numpy as np
import torch
from types import MethodType
from pyannote.core import Annotation,Timeline,Segment
from pyannote.database.util import get_annotated

if torch.backends.cuda.matmul.allow_tf32:
    torch.set_float32_matmul_precision("medium")
pl.seed_everything(42)

def main(parameters):

    preprocessors = {"audio": FileFinder()}
    registry.load_database(parameters.train_parameters["dataset"]["database"])
    dataset = get_protocol(parameters.train_parameters["dataset"]["protocol"])

    print(dataset.stats("train"))
    segmentation_task = PixIT(
        dataset,
        duration=parameters.model["segmentation"]["config"]["duration"],
        batch_size=parameters.model["segmentation"]["config"]["batch_size"],
        max_speakers_per_chunk=parameters.model["segmentation"]["config"][
            "max_num_speakers"
        ],
        separation_loss_weight=parameters.model["segmentation"]["config"]["separation_loss_weight"],
        accumulate_gradient=parameters.model["segmentation"]["config"][
            "accumulate_gradient"
        ],
        num_workers=4,

        losses=parameters.model["segmentation"]["config"]["losses"],
    )
    #batch_norm_params
    monitor, direction = segmentation_task.val_monitor
    params = {
        "ssl_model":parameters.model["segmentation"]["config"]["wavlm_version"]
    }
    parameters.model["inference_params"] = params
    if parameters.model["segmentation"]["name"] == "ToTaToNet2":
        segmentation_model = ToTaToNet2(
            task=segmentation_task,**params
        )
    else:
        segmentation_model = ToTaToNet(
            task=segmentation_task,**params
        )

    def configure_optimizers(self):
        optimizer_wavlm = torch.optim.Adam(self.wavlm.parameters(), lr=1e-5)

        other_params = list(
            filter(lambda kv: "wavlm" not in kv[0], self.named_parameters())
        )
        optimizer_rest = torch.optim.Adam(dict(other_params).values(), lr=parameters.train_parameters["train"]["lr"])


        return [
            optimizer_wavlm,
            optimizer_rest,
        ]

    print(f"Optimizer used : {configure_optimizers}")
    segmentation_model.configure_optimizers = MethodType(
        configure_optimizers, segmentation_model
    )


    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    checkpoint = ModelCheckpoint(
        dirpath=f"results/{parameters.experiment}/models/",
        filename="model",
        monitor="loss/val",
        mode="min",
        save_last=True,
        save_weights_only=False,
        verbose=True,
        enable_version_counter=False
    )


    logger = TensorBoardLogger(f"results/{parameters.experiment}/tb_logs", name="")
    callbacks = [checkpoint, lr_monitor]
    callbacks.append(
        EarlyStopping(monitor="loss/val", mode="min", patience=10, verbose=True)
    )

    trainer = pl.Trainer(
        callbacks=callbacks,
        accelerator="gpu",
        logger = logger,
        devices=1,
        num_nodes=1,
    )
    parameters.save()
    #trainer.validate(segmentation_model)
    trainer.fit(segmentation_model)  # , ckpt_path="last")

def parse_loss(arg):
    res = {}

    for a in arg.split(","):
        key,value = a.split(":")
        res[key] = float(value)
    values = torch.from_numpy(np.array(list(res.values())))
    keys = res.keys()
    if values.sum()!=1:
        values = torch.nn.functional.softmax(values)
        print("Warning ! sum of separation weights != 1, softmax applied could mess everything up")
        print(values)
    for k,v in zip(keys,values):
        res[k]=float(v.item())
    print(res)
    return res
if __name__ == "__main__":
    print("IN the script")
    parser = ArgumentParser()

    parser.add_argument("--name", help="Prefix name for the model")
    parser.add_argument('--losses', type=str)
    parser.add_argument('--model',type=str)
    args = parser.parse_args()
    losses = parse_loss(args.losses)
    print(losses)
    update = {
        "model": {
            "segmentation": {
                "name": args.model,
                "config": {
                    "duration": 5,
                    "max_num_speakers": 3,
                    "batch_size": 8,# batchsize * accumulate_grad
                    "separation_loss_weight":0.5,
                    "accumulate_gradient": 2,
                    #"wavlm_version": "microsoft/wavlm-large",
                    "wavlm_version": "microsoft/wavlm-base-plus",
                    "losses": losses,
                },

            },

            "pipeline": "SeRiouSLy",
        },
        "dataset": {
            "database": "/gpfswork/rech/eie/commun/data/ami/database.yml",
            #"database": "data/database.yml",
            "protocol": "AMI-SDM.SpeakerDiarization.Adaptation",
        },
    }
    parameters = Parameters(train_params=update)
    parameters.set_lr(3e-4)
    parameters.create_experiment(name=args.name)
    parameters.set_gpu(torch.cuda.get_device_name())
    parameters.save(f"results/{parameters.experiment}/config.cfg")
    main(parameters)
