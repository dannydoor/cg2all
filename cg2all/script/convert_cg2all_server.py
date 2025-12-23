#!/usr/bin/env python3
from flask import Flask, jsonify, request, send_file, after_this_request
import numpy as np
import torch
import os
import json
import time
import torch
import dgl
import mdtraj

os.environ["OPENMM_PLUGIN_DIR"] = "/dev/null"

from cg2all.lib.libconfig import MODEL_HOME
from cg2all.lib.libdata import (
    PredictionData,
    create_trajectory_from_batch,
    create_topology_from_data,
    standardize_atom_name,
)
from cg2all.lib.residue_constants import read_coarse_grained_topology
import cg2all.lib.libcg
from cg2all.lib.libpdb import write_SSBOND
from cg2all.lib.libter import patch_termini
import cg2all.lib.libmodel

import warnings

warnings.filterwarnings("ignore")

try:
    from dotenv import load_dotenv
    
    load_dotenv()
except ImportError:
    pass

torch_device = os.getenv("TORCH_DEVICE", "cuda:0")
device = torch.device(
    torch_device if torch.cuda.is_available() and torch_device != "cpu" else "cpu"
)
if device.type == "cuda":
    torch.cuda.empty_cache()
print("Using device: {}".format(device))

model_type = "CalphaBasedModel"
ckpt_fn = MODEL_HOME / f"{model_type}-FIX.ckpt"
if not ckpt_fn.exists():
    cg2all.lib.libmodel.download_ckpt_file(model_type, ckpt_fn, True)

ckpt = torch.load(ckpt_fn, map_location=device)
config = ckpt["hyper_parameters"]
cg_model = cg2all.lib.libcg.CalphaBasedModel

config = cg2all.lib.libmodel.set_model_config(config, cg_model, flattened=False)
model = cg2all.lib.libmodel.Model(config, cg_model, compute_loss=False)
#
state_dict = ckpt["state_dict"]
for key in list(state_dict):
    state_dict[".".join(key.split(".")[1:])] = state_dict.pop(key)
model.load_state_dict(state_dict)
model = model.to(device)
model.set_constant_tensors(device)
model.eval()
# n_proc = int(os.getenv("OMP_NUM_THREADS", 1))
print("Model has been successfully loaded")

app = Flask(__name__)


@app.route('/predict', methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    in_pdb_fn = os.path.abspath("tmp.pdb")
    
    file.save(in_pdb_fn)
    
    return predict_all(in_pdb_fn)


def predict_all(in_pdb_fn):
    timing = {}
    input_s = PredictionData(
        in_pdb_fn,
        cg_model,
        topology_map=None,
        dcd_fn=None,
        radius=config.globals.radius,
        chain_break_cutoff=0.1 * 10.0,
        is_all=False,
        fix_atom=config.globals.fix_atom,
        batch_size=1,
    )
    input_s = dgl.dataloading.GraphDataLoader(
        input_s, batch_size=1, num_workers=1, shuffle=False
    )
    
    t0 = time.time()
    batch = next(iter(input_s)).to(device)
    timing["loading_input"] = time.time() - t0
    #
    t0 = time.time()
    with torch.no_grad():
        R = model.forward(batch)[0]["R"]
    timing["forward_pass"] = time.time() - t0
    #
    timing["writing_output"] = time.time()
    traj_s, ssbond_s = create_trajectory_from_batch(batch, R)
    output = patch_termini(traj_s[0])
    
    out_fn = "tmp_out.pdb"
    output.save(out_fn)
    if len(ssbond_s[0]) > 0:
        write_SSBOND(out_fn, output.top, ssbond_s[0])
    out_fn = os.path.abspath(out_fn)
    
    timing["writing_output"] = time.time() - timing["writing_output"]
    
    if device.type == "cuda":
        if 'batch' in locals(): del batch
        if 'R' in locals(): del R
        
        import gc
        gc.collect()
        
        torch.cuda.empty_cache()
    
    @after_this_request
    def cleanup(response):
        try:
            if os.path.exists(in_pdb_fn): os.remove(in_pdb_fn)
        except Exception as e:
            print(f"Error during cleanup: {e}")
        return response
    
    print(timing)
    
    # download_name is what the user sees, mimetype ensures raw text stream
    return send_file(out_fn, mimetype='text/plain', as_attachment=False)


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5050))
    app.run(host=host, port=port)
