import os
import json

def export_model(model, dirpath, modelname: str, saveparams=False):
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)

    model_filepath = os.path.join(dirpath, modelname)
    model_metadata = dict()
    model_metadata['type'] = model.__class__.__name__
    for idx, layer in enumerate(model.seq):
        model_metadata[f'layer{idx}'] = layer.export_metadata()

    with open(model_filepath, 'wt') as file:
        file.write(json.dumps(model_metadata))

    if saveparams:
        for idx, layer in enumerate(model.seq):
            params_data = layer.export_params()
            for name, params in params_data.items():
                for didx, data in enumerate(params):
                    with open(os.path.join(dirpath, f"{modelname}_layer{idx}_{name}_{didx}"), 'wb') as pfile:
                        pfile.write(data)