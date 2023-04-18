# cover-type-ml-models-classifier

First you need to install necessary packages.
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

After that you can run file models.py. It will create models and visualizations if you do not have them. Be carefull because if you do not have file with hyperparameters, hypertuning will start. If hypertuning stars, you will see training neural-networks with epoch = 5 in your terminal.

```
python models.py
```

Visualizations you can find in figures directory - models compare and neural-network trainging curves.

You can also run simple REST api server
```
python api_REST.py &
```
and next demo, that will show how it works.
```
python demo.py
```

