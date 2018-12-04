
Requirements for running the code:
1. NLTK
2. PyTorch0.4

Create an empty folder `trained_models` in the main directory. This is where trained models will be saved.

Download the MultiNLI (version 0.9) dataset, and put the jsonl files into the folder `nli/data/raw/multinli0.9/`.

**TRAINING ON SOURCE AND 1 TARGET**

To train on a source domain (e.g. fiction), do
`python3 nli/nli_train.py -g fiction`

The `-g` flag specifies which genre you want to train on. There are some other flags you can use, these can be found in `nli/nli_train.py`. The command above will run training and save the model in the folder `trained_models/mnli_xxxxxxxx/`, where `xxxxxxx` is a time-stamp. 

For transfer learning on a different domain, e.g. government, run:
`python3 nli/nli_train.py -tl True -t mnli_xxxxxxx/_x.tar -tg government`

where the `-tl` flag specifies that we wish to do transfer learning, `-t` specifies the model you want to load, and `-tg` is the target genre. The default number of slots is 500. To change this, you can change `self.target_mem_slots` in `nli/encoder.py (line 29)`.

**TESTING**

To evaluate the model performance on the test set of a domain (e.g. government), follow the instructions given in comments of `nli/encoder.py`, make those changes, and then run:
`python3 nli/test_nli.py mnli_xxxxxxx/_x.tar government`

**TRAINING AND TEST ON MULTIPLE TARGET DOMAINS**

Once you have trained on the source and ONE target domain, if you wish to train on a 2nd target domain, you need to change the following hard-coded settings in the code.
1) In `nli/nli_train.py`: Run model.encoder.add_target_pad_2() instead of `model.encoder.add_target_pad()` on `line 95`.
2) In `nli/encoder.py`: In `__init__()`, following the comments to make the necessary changes. Then follow the training and testing instructions given above.
