# Introduction 
This is the implemantation of ComSL, which includes the code for whisper and mbart finetuning, and the code for the ComSL model.

# Getting Started
To run the code, first install the requirements:
```
pip install -r requirements.txt
```

Then change the path to the data or model in the config file config/exp_spec.

After that use command to start training:
```
python3 main.py -c XXX.yaml
```
where XXX.yaml is the configuration file in config/exp_spec.
