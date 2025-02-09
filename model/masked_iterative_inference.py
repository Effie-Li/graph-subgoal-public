from collections import defaultdict
import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from model.transformers import GraphRandomMaskedTransformer, EdgeSearchRandomMaskedTransformer
import utils.analysis

def select_random_token(available_nodes):
    '''
    args
    ----
    available_nodes : torch.tensor, bool, shape (bsz, max_path_len)

    return
    ------
    torch.tensor, bool, shape (bsz, max_path_len), each row has one True
    '''
    selected = torch.multinomial(available_nodes.float(), num_samples=1) # (bsz, 1)
    return F.one_hot(selected[:, 0], num_classes=available_nodes.shape[1]).bool() # (bsz, max_path_len)

def negative_entropy(probs):
    '''
    args
    ----
    probs : torch.tensor, shape (bsz, n, n_node)
    '''
    return (probs * probs.log2()).sum(-1)

def select_max_negent(logits, available_nodes, sample=False, T=1.0):
    '''
    the more peaked a prob dist is, the smaller the entropy, the larger the negent
    select the node with max negent or sample with probability softmax(negent/T)

    args
    ----
    logits : torch.tensor, shape (bsz, n, n_node)
    available_nodes : torch.tensor, bool, shape (bsz, max_path_len)
    sample : bool, whether to sample from the neg ent distribution or take argmax

    return
    ------
    torch.tensor, shape (bsz, n), each row has one True
    '''
    probs = F.softmax(logits, dim=-1)
    negent = negative_entropy(probs)
    negent.masked_fill_(~available_nodes, float('-inf'))
    # check ties
    if len((negent == negent.max(-1)[0].unsqueeze(-1)).nonzero()) > len(negent):
        print('ties found!!!')
    selected = negent.argmax(-1) if not sample else \
               torch.multinomial(utils.analysis.temperature_modulated_softmax(negent, T=T), num_samples=1).squeeze(-1)
    return F.one_hot(selected, num_classes=negent.shape[1]).bool()

def get_token_conditional_logits(model, test_batch, pred_nodes, target_nodes):
    '''
    call clamp_forward() max_n_intermediate_nodes times to mask a node (column)
    and get conditional logits for that token
    '''

    bsz, max_path_len = pred_nodes.shape
    n_node = model.node_embeddings.num_embeddings-1
    conditional_logits = torch.zeros(bsz, max_path_len, n_node).float()
    n_intermediate_nodes = target_nodes.sum(-1)
    for n in range(n_intermediate_nodes.max()):

        # each time masking out tokens in one intermediate position for the entire batch
        token_to_mask = F.one_hot(torch.tensor(n+1), max_path_len).bool()
        run_these_seqs = n < n_intermediate_nodes
        batch_to_run = {k: test_batch[k][run_these_seqs] for k in test_batch}

        # copy over predicted nodes and mask out specific column
        input_nodes = pred_nodes[run_these_seqs].clone() # (unfinished, max_path_len)
        input_nodes.masked_fill_(token_to_mask, model.node_embeddings.padding_idx)

        # generate clamp tensor
        clamp_tensor = model.node_embeddings(input_nodes) - model.output_tokens(torch.tensor(0))
        clamp_tensor.masked_fill_((input_nodes==model.node_embeddings.padding_idx).unsqueeze(-1), 0.0)

        # get conditional logits
        masked_logits = utils.analysis.masked_graphembed_model_clamp_forward(
            model, 
            batch_to_run, 
            path_clamp_tensor=clamp_tensor
        )
        # fill the logits for masked tokens to conditional logits
        conditional_logits[run_these_seqs] = torch.where(
            repeat(token_to_mask, 'n -> b n d', b=masked_logits.shape[0], d=1), 
            masked_logits, 
            conditional_logits[run_these_seqs]
        )

    return conditional_logits # (bsz, max_path_len, n_node)

def select_min_condprob(conditional_logits, input_nodes, available_nodes, sample=False, T=1.0):
    '''
    select or sample the node with the lowest conditional probability

    args
    ----
    logits : torch.tensor, shape (bsz, n, n_node)
    available_nodes : torch.tensor, bool, shape (bsz, max_path_len)
    sample : bool, whether to sample from the conditional prob distribution or take argmin

    return
    ------
    torch.tensor, shape (bsz, n), each row has one True
    '''
    probs = F.softmax(conditional_logits, dim=-1) # (unfinished, max_path_len, n_node)
    # make a temperary tokens var to avoid paded values giving index out of bound error
    tokens = torch.where(available_nodes, input_nodes, 0)
    # index into probs to get token probability for target nodes
    token_probs = probs.gather(dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1) # (unfinished, max_path_len)
    
    # NOTE: make non-target nodes have prob=inf. prob=1.0 is not enough 
    # because the output prob becomes so peaked that with the float precision 
    # it effectively becomes p(t)=1.0 and armin will forever select the firs token
    # and the inference loop gets stuck. inf to support sampling after softmax
    token_probs.masked_fill_(~available_nodes, float('inf'))
    if len((token_probs == token_probs.min(-1)[0].unsqueeze(-1)).nonzero()) > len(token_probs):
        print('ties found!!!')
    selected = token_probs.argmin(-1) if not sample else \
               torch.multinomial(utils.analysis.temperature_modulated_softmax(-token_probs, T=T), num_samples=1).squeeze(-1)
    return F.one_hot(selected, num_classes=tokens.shape[1]).bool()

def iterative_inference_loop(model, test_batch, sample_mode, force_new_node=True, node_sample_T=1.0, max_iter=10, debug_verbose=False):

    '''
    args
    ----
    sample_mode : str, 'min_condprob' or 'sample_condprob or 'max_negent' or 'sample_negent' or 'random'
    '''

    # all seqs start as fully-masked. 
    # we need to loop at least n_intermediate_node times for each seq, potentially more 
    # if we're sampling nodes instead of using argmax, each time selecting one node to predict
    # method 1: randomly pick an unpredicted intermediate node
    # metohd 2 & 3: adaptively -- use negative entropy or token prob
    
    # TODO: need to take care of pred order when force_new_node=False

    # # for debugging outside of function purposes
    # sample_mode = 'sample_condprob'
    # max_iter = 10
    # force_new_node = True
    # debug_verbose = True

    if debug_verbose and len(test_batch['index']) > 10:
        print('...careful of batch size while debugging...')
        return
    
    assert type(model) in [GraphRandomMaskedTransformer, EdgeSearchRandomMaskedTransformer]
    model.mask_all = True
    
    assert sample_mode in ['random', 'max_negent', 'sample_negent', 'min_condprob', 'sample_condprob']

    bsz = len(test_batch['index'])
    finished = torch.zeros(bsz).bool()
    _ = model.input_embed(test_batch) # lazy hack so we have model.batch_output_mask
    target_nodes = model.batch_output_mask.clone()
    remaining_nodes = target_nodes.clone()
    pred_nodes = torch.ones(bsz, model.max_path_len).long() * model.node_embeddings.padding_idx
    pred_order = torch.ones(bsz, model.max_path_len).long() * -1
    clamp_tensor = torch.zeros(bsz, model.max_path_len, model.embed_dim)

    if sample_mode in ['min_condprob', 'sample_condprob']:
        # initialize lowprob sampling by first generate all target tokens once
        logits = utils.analysis.masked_graphembed_model_clamp_forward(
            model, 
            test_batch,
            path_clamp_tensor=clamp_tensor
        )
        pred_nodes[target_nodes] = logits.argmax(-1)[target_nodes]

    round = 0
    while sum(finished==0) > 0 and round < max_iter:

        if debug_verbose: print('round:', round)

        batch_to_run = {k:test_batch[k][~finished] for k in test_batch}
        logits = utils.analysis.masked_graphembed_model_clamp_forward(
            model, 
            batch_to_run, 
            path_clamp_tensor=clamp_tensor[~finished]
        ) # (n_unfinished, max_path_len, n_node)
        
        available_nodes = remaining_nodes[~finished] if force_new_node else target_nodes[~finished]

        if debug_verbose: print('available_nodes:', available_nodes)
        
        # randomly pick an intermediate node -> bool mask (n_unfinished, max_path_len)
        if sample_mode == 'random':
            selected = select_random_token(available_nodes=available_nodes)
        elif sample_mode == 'max_negent':
            selected = select_max_negent(logits, available_nodes=available_nodes, sample=False)
        elif sample_mode == 'sample_negent':
            selected = select_max_negent(logits, available_nodes=available_nodes, sample=True, T=node_sample_T)
        elif sample_mode in ['min_condprob', 'sample_condprob']:
            conditional_logits = get_token_conditional_logits(
                model, batch_to_run, pred_nodes[~finished], target_nodes[~finished]
            )
            selected = select_min_condprob(
                conditional_logits, pred_nodes[~finished], available_nodes=available_nodes,
                sample=True if sample_mode == 'sample_condprob' else False, T=node_sample_T,
            )
            # override logits so we use conditional logits for updating the next token
            logits = conditional_logits
        
        if debug_verbose: print('selected:', selected)

        # udpate pred_nodes
        nodes = logits[selected].argmax(-1).unsqueeze(-1) # (n_unfinished, 1)
        pred_nodes[~finished] = torch.where(selected, nodes, pred_nodes[~finished])
        pred_order[~finished] = torch.where(selected, torch.tensor(round), pred_order[~finished])

        if debug_verbose: print('pred_nodes:', pred_nodes)
        if debug_verbose: print('pred_order:', pred_order)
        
        # update clamp tensor
        # NOTE: we subtract model.output_token embedding to offset the model's default input_embed
        # so that for clamped tokens the resulting input embed is node_embed + rel_pos_to_start + rel_pos_to_goal
        new_token_embeddings = model.node_embeddings(nodes) - model.output_tokens(torch.tensor(0)) # (n_unfinished, 1, embed_dim)
        clamp_tensor[~finished] = torch.where(repeat(selected, 'b n -> b n d', d=model.embed_dim), 
                                            new_token_embeddings, 
                                            clamp_tensor[~finished])

        if debug_verbose: print('clamp_tensor:', clamp_tensor.sum(-1))
        
        # update remaining_nodes
        remaining_nodes[~finished] = torch.logical_and(remaining_nodes[~finished], ~selected)
        finished = remaining_nodes.sum(-1) == 0

        if debug_verbose: print('remaining_nodes:', remaining_nodes)
        if debug_verbose: print('finished:', finished)

        round += 1

    return pred_nodes, pred_order, target_nodes, finished

def pred_tensor_to_dict(pred_nodes, target_nodes, test_batch, datamodule):
    # turn predicted nodes into a result dict: probe_idx -> (pred_path, target_paths, acc) mapping
    predictions = {}
    for i, probe in enumerate(test_batch['index']):
        probe = probe.item()
        pred_path = pred_nodes[i][:test_batch['path_len'][i]]
        target_paths = datamodule.dataset.raw_data['shortest_path'][probe]
        m = target_nodes[i][:test_batch['path_len'][i]]
        acc = [(pred_path[m] == torch.tensor(p)[m]).float().mean().item() for p in target_paths]
        predictions[probe] = (pred_path.tolist(), target_paths, acc)
    return predictions

def pred_node_order_to_df(pred_nodes, pred_order, target_nodes, correct, batch, batch_metadata):
    # turn intermediate nodes into a dataframe recording the order of prediction
    df = defaultdict(list)
    for batch_i, probe in enumerate(batch['index']):
        probe = probe.item()
        nodes = pred_nodes[batch_i][target_nodes[batch_i]].tolist()
        order = pred_order[batch_i][target_nodes[batch_i]].tolist()
        df['graph_context'].extend([batch['graph_context'][batch_i].item()]*len(nodes))
        df['probe'].extend([probe]*len(nodes))
        df['path_len'].extend([batch['path_len'][batch_i].item()]*len(nodes))
        df['seq_correct'].extend([correct[batch_i].item()]*len(nodes))
        df['pred_node'].extend(nodes)
        df['pred_order'].extend(order)
        df['node_pos'].extend(torch.where(target_nodes[batch_i])[0].tolist())
        df['node_BC'].extend([batch_metadata['graph_BC_scores'][probe][n] for n in nodes])
        df['node_DC'].extend([batch_metadata['graph_DC_scores'][probe][n] for n in nodes])
    df = pd.DataFrame(df)
    return df