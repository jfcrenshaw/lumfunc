# lumfunc

Installation (from the root directory):

```bash
mamba create -n lumfunc
mamba activate lumfunc
pip install -e .
python -m ipykernel install --user --name lumfunc
```

You can create the photometric library using a command line like

```bash
nohup nice python bin/create_libraries.py test &
```
