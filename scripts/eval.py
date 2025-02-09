import dill as pkl
import torch
from model.model_module import (
    GraphAutoregTransformer, 
    GraphMaskedTransformer, 
    GraphRandomMaskedTransformer,
    EdgeSearchAutoregTransformer, 
    EdgeSearchRandomMaskedTransformer,
)

def model_eval(model, datamodule, device, fname):
    '''
    get model prediction and accuracy for all validation problems (accuracies compared to each shortest path)
    and cache result
    '''

    datamodule.batch_size = len(datamodule.val_idx)
    for val_batch in datamodule.val_dataloader(): break
    model = model.to(device)
    val_batch = {k:val_batch[k].to(device) for k in val_batch}
    result = {} # probe_idx -> (pred_path, target_paths, acc) mapping

    visited = set()
    for batch_idx, probe_idx in enumerate(val_batch['index']):
        probe_idx = probe_idx.item()
        if probe_idx in visited: continue
        visited.add(probe_idx)

        target_paths = datamodule.dataset.raw_data['shortest_path'][probe_idx]
        mini_batch = {k:val_batch[k][batch_idx:batch_idx+1] for k in val_batch}

        if type(model.transformer) in [GraphAutoregTransformer, EdgeSearchAutoregTransformer]: # always evaluate autoreg models with rollout
            with torch.no_grad():
                out, _, _ = model.transformer.forward_rollout(mini_batch)
                result_dict = model.transformer.generate_result_dict(mini_batch, out)
            pred_path = result_dict['pred'][0].argmax(-1)[:len(target_paths[0])].to('cpu') # will cut it at number of edge needed
            acc = [(torch.tensor(p) == pred_path).float().mean().item() for p in target_paths] # compare to each path
        elif type(model.transformer) in [GraphMaskedTransformer, GraphRandomMaskedTransformer, EdgeSearchRandomMaskedTransformer]:
            with torch.no_grad():
                result_dict, _, _ = model.forward(mini_batch)
            pred_path = result_dict['pred'][0].argmax(-1)[:len(target_paths[0])].to('cpu') # will cut it at number of edge needed
            output_mask = model.transformer.batch_output_mask[0][:len(target_paths[0])]
            acc = [(torch.tensor(p)[output_mask] == pred_path[output_mask]).float().mean().item() for p in target_paths] # compare to each path

        result[probe_idx] = (pred_path.numpy().tolist(), target_paths, acc)

    if fname is not None:
        pkl.dump(result, open(fname, 'wb'))
    else:
        return result