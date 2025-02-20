# Global imports
import os
from argparse import ArgumentParser
from rich.progress import track
from rich.console import Console
from rich.table import Table
import traceback
import glob
import pickle as pkl
import numpy as np
import decimal

# ML imports
import torch
import torchaudio
from meeteval.io import STM
import whisperx
from whisper.normalizers import EnglishTextNormalizer

# Pyannote imports
from pyannote.database.util import load_rttm, load_uem
from pyannote.database import get_protocol, FileFinder
from pyannote.database.registry import registry
from pyannote.audio.pipelines import SpeechSeparation as SeRiouSLy
from pyannote.audio import Pipeline, Model
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.core import Segment,Timeline,Annotation

# Local imports
from src.cp_wer import cp_word_error_rate_partition
from src.xml import XML
normalizer = EnglishTextNormalizer()
compute_type = "float32"  # "float16"
modelx = whisperx.load_model("large-v3", device="cuda", compute_type=compute_type)
model_a, metadata = whisperx.load_align_model(language_code="en", device="cuda")

def apply_whisperx(audio, sample_rate=16000, fname=None, spkid="A"):
    audio = np.float32(audio / np.max(np.abs(audio)))
    result = modelx.transcribe(audio, batch_size=8, language="en")
    result2 = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        "cuda",
        return_char_alignments=False,
    )
    output = result2["segments"]  # after alignment
    text_ts = ""
    prev_end = 0
    word_buf = []
    flag = False
    for words in output:
        for word in words["words"]:
            if flag and "start" in word:
                stop = word["start"]
                start = prev_end
                txt = " ".join(word_buf)
                text_ts += f"{fname} 0 {spkid} {start} {stop} {normalizer(txt)}\n"
                # print(f"{fname} 0 {spkid} {start} {stop} {normalizer(txt)}\n")
                flag = False
                word_buf = []
            if "start" not in word:
                # print(word)
                word_buf.append(word["word"])
                flag = True
                continue

            text_ts += f"{fname} 0 {spkid} {word['start']} {word['end']} {normalizer(word['word'])}\n"
            prev_end = word["end"]

    out = STM.parse(text_ts).to_seglst()
    return out


def eval_separation(sources, references, session_id, partition):
    text = None
    for s in range(sources.shape[1]):
        source = sources[:, s]
        txt = apply_whisperx(source, fname=session_id, spkid=s)
        if text is None:
            text = txt
        else:
            text += txt
    
    cpwer_dict = cp_word_error_rate_partition(references, text, partition)
    return cpwer_dict
console = Console(record=True)
parser = ArgumentParser()
parser.add_argument(
    "--database",
    default="/path/to/pyannotedb/database.yml",
    help="Path to the pyannote database, rather should be as an environment variable",
)
parser.add_argument(
    "--protocol",
    default=["AMI-SDM.SpeakerDiarization.mini"],
    nargs="+",
    help="Protocol to use for training",
)
parser.add_argument("--seg_th", default=0.5, help="Segmentation threshold")
parser.add_argument("--clu_th", default=0.68, help="Clustering threshold")
parser.add_argument("--min_cluster_size", default=50, help="Minimum size of a cluster")

parser.add_argument("--model_cfg", help="path to the config of the model to evaluate")

args = parser.parse_args()



def apply_pipeline(max_speakers):



    check_audio = True
    eval_sep = True
    acc_D = {1:0,0:0,"control":0}
    acc_S = {1:0,0:0,"control":0}
    acc_C = {1:0,0:0,"control":0}
    acc_I = {1:0,0:0,"control":0}
    metrics = {1:{},0:{},"control":{}}
    for file in track(files):
        uri = file["uri"]
        # REF for DIAR
        diar_ref: Annotation = file['annotation']

        diar_ov:Timeline = diar_ref.get_overlap()
        diar_nov:Timeline = diar_ref.extrude(diar_ov).get_timeline()

        ref_ov:Annotation = diar_ref.crop(diar_ov)
        ref_nov:Annotation = diar_ref.crop(diar_nov)

        ref = {1:ref_ov,0:ref_nov,'control':diar_ref}
        print(f"DURATION REF OV : {diar_ov.duration()}, REF NOV : {diar_nov.duration()} ")

        # REF for ASR

        xml_files = glob.glob(f"../data_tmp/AMI-diarization-setup/words/{uri}.*.xml") #standard AMI words

        xml = XML.load(xml_files, parse_float=decimal.Decimal)

        ref_asr = xml.to_seglst()
        
        # Processing
        print(f"Processing file {file['uri']}", flush=True)
        try:
            diarization, sources = pipeline.apply(file, max_speakers=max_speakers)
        except Exception as e:
            print("WARNING FAIL")
           
            continue
        # Separation for ASR
        cpwer = eval_separation(sources.data,ref_asr,session_id=uri,partition=diar_ov)
        if cpwer is None:
            print(ref_asr)
            continue
        diarization_ov = diarization.crop(diar_ov)
        diarization_nov = diarization.crop(diar_nov)
        diarization = {1:diarization_ov,0:diarization_nov,'control':diarization}
        for part in cpwer:
            metrics[part][uri] = {"wer":None,"insertions":None,"deletions":None,"substitutions":None}
            print(f"RESULT FOR PARTITION {part} of file {uri}")

            wer = cpwer[part]
            deletion_rate = wer.deletions / wer.length * 100
            insertion_rate = wer.insertions / wer.length * 100
            substitution_rate = wer.substitutions / wer.length * 100
            wer_correct = wer.length - wer.errors
            acc_C[part] += wer_correct
            acc_S[part] += wer.substitutions
            acc_D[part] += wer.deletions
            acc_I[part] += wer.insertions

            print(f"Substitution rate: {substitution_rate:.1f}%")
            print(f"Deletion rate: {deletion_rate:.1f}%")
            print(f"Insertion rate: {insertion_rate:.1f}%")
            
            print(f"WER for file {uri} : {wer.error_rate * 100:.1f}%")
            metrics[part][uri]["wer"] = wer.error_rate * 100
            metrics[part][uri]["insertions"]=insertion_rate
            metrics[part][uri]["deletions"]=deletion_rate
            metrics[part][uri]["substitutions"]=substitution_rate

            # evaluate its performance
            diar_result_current = metric(
                ref[part], diarization[part], detailed=True
            )
            FA = diar_result_current["false alarm"] / diar_result_current["total"] * 100
            MD = (
                diar_result_current["missed detection"] / diar_result_current["total"] * 100
            )
            SC = diar_result_current["confusion"] / diar_result_current["total"] * 100

            print(f"Diar res for part {part} of {uri}")
            print(f"False alarm: {FA:.1f}")
            print(f"Missed detection: {MD:.1f}")
            print(f"Speaker confusion: {SC:.1f}")
            print(f"Total DER: {FA+MD+SC:.1f}")
            metrics[part][uri]["der"] = diar_result_current["false alarm"]+diar_result_current["missed detection"]+diar_result_current["confusion"]
            metrics[part][uri]["miss"]=diar_result_current["missed detection"]
            metrics[part][uri]["fa"]=diar_result_current["false alarm"]
            metrics[part][uri]["confusion"]=diar_result_current["confusion"]
            metrics[part][uri]["total"] = diar_result_current["total"]
    # Global metrics
    for part in [1,0,"control"]:
        deletion_rate = acc_D[part] / (acc_C[part] + acc_S[part] + acc_D[part]) * 100
        insertion_rate = acc_I[part] / (acc_C[part] + acc_S[part] + acc_D[part]) * 100
        substitution_rate = acc_S[part] / (acc_C[part] + acc_S[part] + acc_D[part]) * 100
        metrics[part]["all"] = {}
        metrics[part]["all"]["wer"] = insertion_rate + deletion_rate + substitution_rate
        metrics[part]["all"]["insertions"]=insertion_rate
        metrics[part]["all"]["deletions"]=deletion_rate
        metrics[part]["all"]["substitutions"]=substitution_rate
        acc_total = sum(metrics[part][uri]["total"] for uri in metrics[part] if uri != "all")
        acc_fa = sum(metrics[part][uri]["fa"] for uri in metrics[part] if uri != "all")
        acc_miss = sum(metrics[part][uri]["miss"] for uri in metrics[part] if uri != "all")
        acc_conf = sum(metrics[part][uri]["confusion"] for uri in metrics[part] if uri != "all")
        total = acc_total
        metrics[part]["all"]["total"] = total
        metrics[part]["all"]["der"]=acc_miss+acc_fa+acc_conf
        metrics[part]["all"]["miss"]=acc_miss
        metrics[part]["all"]["fa"]=acc_fa
        metrics[part]["all"]["confusion"]=acc_conf
    return metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(args.protocol)
if "PYANNOTE_DATABASE_CONFIG" not in os.environ:
    # Pyannote database should be as an environment variable
    os.environ["PYANNOTE_DATABASE_CONFIG"] = args.database
console.print(
    "[green]PYANNOTE_DATABASE_CONFIG[/] :", os.environ["PYANNOTE_DATABASE_CONFIG"]
)
registry.load_database(os.environ["PYANNOTE_DATABASE_CONFIG"])

datasets = [get_protocol(prot, {"audio": FileFinder()}) for prot in args.protocol]
# Loading best model
pipeline_bs = 8
max_speakers = 5
if args.model_cfg == "results/baseline/config.cfg":
    pipeline = Pipeline.from_pretrained(
        "pyannote/speech-separation-ami-1.0"
    )
    pipeline.instantiate(
        {
            "segmentation": {"min_duration_off": 0.0, "threshold": 0.5},
            "clustering": {
                "method": "centroid",
                "min_cluster_size": 50,
                "threshold": 0.68,
            },
            "separation": {
                "leakage_removal": True,
                "asr_collar": 0.32,
            }
        }
    )
    pipeline = pipeline.to(device)
else:

    model_ckpt = "model_checkpoint"
    params = {}#model parameters


    finetuned_model = Model.from_pretrained(
        model_ckpt, device=device, local_files_only=True, **params
    )
    finetuned_model = finetuned_model.to(device)

    pipeline = SeRiouSLy(
        segmentation=finetuned_model,
        segmentation_batch_size=pipeline_bs,
        embedding_batch_size=pipeline_bs,
        embedding_exclude_overlap=False,  # culprit
        verbose=False
    ).to(device)
    asr_collar = 0.32
    pipeline.instantiate( # hyperparameters instantiation
        {
            "segmentation": {
                "min_duration_off": 0.0,
                "threshold": 0.5,
            },
            "clustering": {
                "method": "centroid",
                "min_cluster_size": 50,
                "threshold": 0.5,
            },
            "separation": {
                "leakage_removal": True,
                "asr_collar": asr_collar,
            },
        }
    )


metric = DiarizationErrorRate()
eval_param = {
    "dataset": {
        "database": os.environ["PYANNOTE_DATABASE_CONFIG"],
        "protocol": args.protocol,
        "n_files": [d.stats("test")["n_files"] for d in datasets],
    },
    "metric": metric.name,
}


for protocol in datasets:
    metric.reset()
    check_audio = True
    files = list(protocol.test())
    metrics = apply_pipeline(
        max_speakers=max_speakers,
    )
    for part in [1,0,'control']:
        table_wer = Table(title=f"CPWER Evaluation Results {protocol.name} for part {part}")
        table_wer.add_column("uri")
        table_wer.add_column("CPWER")
        table_wer.add_column("Deletion rate %")
        table_wer.add_column("Insertion rate %")
        table_wer.add_column("Substitution rate %")
        for uri in metrics[part]:
            m = metrics[part][uri]
            table_wer.add_row(
                uri,
                f"{m['wer']:.2f}",
                f"{m['deletions']:.2f}",
                f"{m['insertions']:.2f}",
                f"{m['substitutions']:.2f}",
            )

        table = Table(title=f"DER Evaluation Results {protocol.name} for part {part}")
        table.add_column("uri")
        table.add_column("DER")
        table.add_column("total")
        table.add_column("False Alarm %")
        table.add_column("Miss %")
        table.add_column("Confusion %")
        for uri in metrics[part]:
            m = metrics[part][uri]
            table.add_row(
                uri,
                f"{m['der']/m['total']*100:.2f}",
                f"{m['total']}",
                f"{m['fa']/m['total']*100:.2f}",
                f"{m['miss']/m['total']*100:.2f}",
                f"{m['confusion']/m['total']*100:.2f}",
            )

        console.print(table)
        console.print(table_wer)
        # console.print(metric)

    console.print()
