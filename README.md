# MSciProject
Code Base for my MSci Project

To create env:
`conda env create -f environment.yml`

conda config --append channels conda-forge

`pip install -r requirements.txt`


`cd RecBolePyTorch`
`pip install -e . --verbose`
`pip install ray>0.7.5`
`pip install pandas, tabulate, torch`

`python run_recbole.py --epochs=30 --model=GRU4Rec --neg_sampling=None --train_neg_sample_args=None --log_wandb=False --use_gpu=True`

`python run_recbole.py --epochs=30 --model=BPR --log_wandb=True`

Train GRU 
`python run_recbole.py --epochs=10 --model=GRU4Rec --neg_sampling=None --train_neg_sample_args=None --log_wandb=False --use_gpu=True`

We use labels for skipped to represent negative items. 
Run the following command for the different models and loss functions

Models to Use:
**
 could use all sequential recommender systems. Use most common and some which could possibly perform best 
**
GRU4Rec
BERT4Rec
SASRec


`python run_recbole.py --epochs=10 --model={model} --neg_sampling=None --log_wandb=True --use_gpu=True --loss_type={loss}`



HYPER TEST

python3 run_hyper.py --config_files=['recbole/properties/]--dataset=lfm-100k --model=GRU4Rec --loss=BPR
