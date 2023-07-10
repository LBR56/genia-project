from Preprocess.Preprocess import dutosec
from aws import AwsController
from Model.model import NER, Formal
from process.process import Formalprocess, NERprocess, exist_label, NER_ratio
from process.visualization import Formal_plot, NERratio_plot, Wordbysecond_plot

def process_1(ckpt_path,conf_path) :
    aws_controller = AwsController()
    df = aws_controller.s3_download(["preprocessed.csv"], "src/")["preprocessed"]
