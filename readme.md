# Code implementation for LLM4EA in NeurIPS-2024 submission

### Quick Start
**Step1.** Instanll the required packages by running the following command:
```
pip install -r requirements.txt
```

**Step2.** Download the dataset from [here](https://anonymous.4open.science/r/processedOpenEAData-3674/) and put it in the `data` folder.

**Step2.** Specify the `gpt-api-key` in the `config.py` file with your openai API key.

**Step3.** Run the following command to run llm4ea on D-Y-15k dataset
```
python infer.py --dataset_name D-Y-15k
```

### Simulations
If you have no access to an OpenAI API, you can run the simulation by running the following command:
```
python infer.py --dataset_name D-Y-15k --simulate --tpr 0.5
```
here, the arguement `--tpr` specifies the true positive rate for the synthesized pseudo-labels.


### Ablation settings

There are two other optional scripts `infer-active-only.py` and `infer-lr-only.py`. which is the variant of the `infer.py` script. 
- The `infer-active-only.py` script deactivates the label refinement component of the model, 
- and the `infer-lr-only.py` script deactivates the active learning component of the model. 
These corresponds to the `w/o LR` and `w/o Act` ablation settings in the paper.


### Acknowledgement
The code is based on [PRASE](https://github.com/qizhyuan/PRASE-Python) and [Dual-AMN](https://github.com/MaoXinn/Dual-AMN), the dataset is from [OpenEA benchmark](https://github.com/nju-websoft/OpenEA), preprocessed by using the dump file `wikidatawiki-20160801-abstract.xml` from [wikdata](https://archive.org/download/wikidatawiki-20160801).