from collections import defaultdict
import torch
import numpy as np
import sklearn
from sklearn import linear_model
import networkx as nx
import torch.nn.functional as F

from data.oracle import FindPath
from model.transformers import GraphAutoregTransformer, GraphMaskedTransformer

def get_edge_betweenness_centrality(edges):
    g = nx.from_edgelist(edges)
    values = nx.edge_betweenness_centrality(g)
    values = {tuple(e): values[tuple(e)] if tuple(e) in values else values[tuple(reversed(e))] 
              for e in edges}
    return values

def get_node_betweenness_centrality(edges):
    g = nx.from_edgelist(edges)
    values = nx.betweenness_centrality(g)
    return values

def get_node_degree_centrality(edges):
    g = nx.from_edgelist(edges)
    values = nx.degree_centrality(g)
    return values

def softmax(x):
    return np.exp(x) / sum(np.exp(x))

def generate_filler(g, state_visitation, n_filler):

    '''
    creates some filler trials to increase visitation to states with low BC score
    samples start/goal pairs inversely proportional to current state visitaition count
    keep len1 and len2 problems and problems that don't contain a path passing top2 visited nodes
    (both heuristics are trying to increase visitation to infreqly visited states; 
    validated by some pilot tests)
    '''

    filler_dataset = defaultdict(list)

    while n_filler > 0:

        # use state visitation to generate sampling prob
        sampling_prob = 1 - state_visitation / sum(state_visitation)
        sampling_prob = softmax(sampling_prob * 100) # scaling param roughly empirically searched for
        
        # sample problem
        start, goal = np.random.choice(range(8), size=2, p=sampling_prob, replace=False)
        shortest_paths = FindPath().solve(g, start, goal, mode='best')

        # filter problem
        infreq_rank = np.argsort(state_visitation).tolist()
        if max([infreq_rank.index(n) for n in np.unique(shortest_paths)]) >= 6: 
            continue # skip problems with top2 visited nodes
        
        n_filler -= 1

        for path in shortest_paths:
            for n in path:
                state_visitation[n] += 1

        filler_dataset['start_goal'].append((start, goal))
        filler_dataset['shortest_path'].append(shortest_paths)
        filler_dataset['path_length'].append(len(shortest_paths[0]))
    
    return filler_dataset

def generate_batch_metadata(dm, batch):
    probe_indices = []
    n_shortest_path = {} # at the probe instance level
    path_BC_scores = {} # at the probe instance level
    graph_BC_scores = {} # at the probe instance level
    path_DC_scores = {} # at the probe instance level
    graph_DC_scores = {} # at the probe instance level

    for probe_ind in batch['index']:
        probe_ind = probe_ind.item()
        probe_indices.append(probe_ind)

        # 1. how many multi-shortest paths are there for this problem
        n_shortest_path[probe_ind] = len(dm.dataset.raw_data['shortest_path'][probe_ind])

        # 2. what is the BC score for all nodes in this graph
        if probe_ind in path_BC_scores: continue
        edges = dm.dataset.raw_data['edge_list'][probe_ind]
        BC_scores = get_node_betweenness_centrality(edges)
        graph_BC_scores[probe_ind] = BC_scores

        # 3. what is the BC score of all intermediate nodes in this path
        shortest_paths = dm.dataset.raw_data['shortest_path'][probe_ind]
        avg_BC = [np.mean([BC_scores[n] for n in path[1:-1]]) for path in shortest_paths]
        max_BC = [np.max([BC_scores[n] for n in path[1:-1]]) for path in shortest_paths]
        path_BC_scores[probe_ind] = {'avg': avg_BC, 'max': max_BC}

        # 4. what is the DC score for all nodes in this graph
        DC_scores = get_node_degree_centrality(edges)
        graph_DC_scores[probe_ind] = DC_scores

        # 5. what is the DC score of all intermediate nodes in this path
        avg_DC = [np.mean([DC_scores[n] for n in path[1:-1]]) for path in shortest_paths]
        max_DC = [np.max([DC_scores[n] for n in path[1:-1]]) for path in shortest_paths]
        path_DC_scores[probe_ind] = {'avg': avg_DC, 'max': max_DC}

    # single-answer problems
    single_answer_problems = np.array([n_shortest_path[i] for i in probe_indices]) == 1

    # multi-answer but same BC score problems
    no_BCpreference_problems = np.array(
        [len(path_BC_scores[i]['avg'])>1 and min(path_BC_scores[i]['avg'])==max(path_BC_scores[i]['avg']) 
        for i in probe_indices]
    )

    # multi-answer but same DC score problems
    no_DCpreference_problems = np.array(
        [len(path_DC_scores[i]['avg'])>1 and min(path_DC_scores[i]['avg'])==max(path_DC_scores[i]['avg']) 
        for i in probe_indices]
    )

    return {'single_answer_problems': single_answer_problems,
            'no_BCpreference_problems': no_BCpreference_problems,
            'no_DCpreference_problems': no_DCpreference_problems,
            'path_BC_scores': path_BC_scores,
            'path_DC_scores': path_DC_scores,
            'graph_BC_scores': graph_BC_scores,
            'graph_DC_scores': graph_DC_scores,
            }

def run_model_over_batch(batch, model, model_type, raw_shortest_paths):

    # TODO: this may need to be tweaked to have more genera use cases
    #       e.g. consider returning a predictions dict {probe_idx: (pred_path, target_paths, acc)}
    #       to interface with other analysis functions

    correct = torch.zeros(len(batch['index'])).bool()
    # get model prediction
    with torch.no_grad():
        if model_type == 'autoreg':
            out, attn_weights, token_reps = model.transformer.forward_rollout(batch, return_attn=True, return_reps=True)
            result_dict = model.transformer.generate_result_dict(batch, out)
        elif model_type == 'masked' or model_type == 'random_masked':
            out, attn_weights, token_reps = model.transformer.get_attn_and_reps(batch)
            result_dict = model.transformer.generate_result_dict(batch, out)
    pred_paths = result_dict['pred'].argmax(-1).to('cpu')
    # compare to candidate paths
    closest_target = batch['shortest_path'].clone()
    for i in range(len(batch['index'])):
        target_paths = raw_shortest_paths[batch['index'][i]]
        pred_path = pred_paths[i][:len(target_paths[0])]
        if model_type == 'autoreg':
            acc = [(torch.tensor(p) == pred_path).float().mean().item() for p in target_paths]
        if model_type == 'masked' or model_type == 'random_masked': 
            m = model.transformer.batch_output_mask[i][:len(target_paths[0])]
            acc = [(torch.tensor(p)[m] == pred_path[m]).float().mean().item() for p in target_paths]
        correct[i] = max(acc) == 1.0
        closest_target[i][:batch['path_len'][i]] = torch.tensor(target_paths[np.argmax(acc)])
    
    return correct, pred_paths, closest_target, attn_weights, token_reps

def accuracy_profile(all_predictions, batch, batch_metadata, verbose=True):
    accs = {} # token-level accuracy for for each problem
    sa_accs = [] # token-level accuracy for problems with a single answer
    ma_accs = [] # token-level accuracy for problems with multiple answers
    equally_wrong = [] # problems where the model is equally wrong against all answers)
    single_answer_problems = batch_metadata['single_answer_problems']

    for probe_idx in all_predictions:
        if probe_idx not in batch['index']: continue
        batch_idx = batch['index'].tolist().index(probe_idx)

        pred_path, target_paths, acc = all_predictions[probe_idx]
        accs[(batch_idx, probe_idx)] = max(acc)

        if single_answer_problems[batch_idx]:
            sa_accs.append(max(acc))
        else:
            ma_accs.append(max(acc))

            if sum(np.array(acc)==acc[0]) == len(acc):
                # the model was equally wrong against all answers; didn't "settle" on any particular path
                equally_wrong.append(probe_idx)
    
    if verbose:
        print(f'% overall acc = 1: {(np.array(list(accs.values()))==1).mean()}')
        print(f'% single_answer acc = 1: {(np.array(sa_accs)==1).mean()}')
        print(f'% multi_answer acc = 1: {(np.array(ma_accs)==1).mean()}')
        print(f'% equally wrong: {len(equally_wrong)/len(ma_accs)}')
    
    return {'acc': accs,
            'sa acc': (np.array(sa_accs)==1).mean(),
            'ma acc': (np.array(ma_accs)==1).mean(),
            'eq wrong': len(equally_wrong)/len(ma_accs)}

def path_preference(all_predictions, batch, batch_metadata, verbose=True):

    single_answer_problems = batch_metadata['single_answer_problems']
    no_BCpreference_problems, no_DCpreference_problems = batch_metadata['no_BCpreference_problems'], batch_metadata['no_DCpreference_problems']
    path_BC_scores, path_DC_scores = batch_metadata['path_BC_scores'], batch_metadata['path_DC_scores']

    problem_type, path_choice_labels = [], []
    BC_baseline, DC_baseline = [], []

    for probe_idx in all_predictions:
        if probe_idx not in batch['index']: continue
        batch_idx = batch['index'].tolist().index(probe_idx)

        pred_path, target_paths, acc = all_predictions[probe_idx]
        if single_answer_problems[batch_idx]: continue

        path_BCs = np.array(path_BC_scores[probe_idx]['avg'])
        path_DCs = np.array(path_DC_scores[probe_idx]['avg'])
        p = np.argmax(acc) # index of model-selected path

        # label problem type and compute baselines
        if no_BCpreference_problems[batch_idx] and no_DCpreference_problems[batch_idx]:
            # no preference to evaluate against
            problem_type.append('no_pref')
        
        elif (not no_BCpreference_problems[batch_idx]) and no_DCpreference_problems[batch_idx]:
            # there is pathBC difference but no pathDC difference
            problem_type.append('BC_only')
            BC_baseline.append((path_BCs==max(path_BCs)).sum()/len(path_BCs))

        elif no_BCpreference_problems[batch_idx] and (not no_DCpreference_problems[batch_idx]):
            # there is pathDC difference but no pathBC difference
            problem_type.append('DC_only')
            DC_baseline.append((path_DCs==max(path_DCs)).sum()/len(path_DCs))
        
        else: # there is both pathBC and pathDC difference
            best_BC_paths = np.where(path_BCs==max(path_BCs))[0]
            best_DC_paths = np.where(path_DCs==max(path_DCs))[0]
            if len(np.intersect1d(best_BC_paths, best_DC_paths)) > 0: # pathBC and pathDC agrees on best path
                problem_type.append('BCDC_agree')
            else:
                problem_type.append('BCDC_disagree')
            BC_baseline.append((path_BCs==max(path_BCs)).sum()/len(path_BCs))
            DC_baseline.append((path_DCs==max(path_DCs)).sum()/len(path_DCs))

        # label model path choice
        if max(acc) != 1:
            path_choice_labels.append('wrong')
        elif problem_type[-1] == 'no_pref':
             path_choice_labels.append('correct_nopref')
        elif problem_type[-1] == 'BC_only': 
            path_choice_labels.append('maxBC' if path_BCs[p] == max(path_BCs) else 'other')
        elif problem_type[-1] == 'DC_only': 
            path_choice_labels.append('maxDC' if path_DCs[p] == max(path_DCs) else 'other')
        
        elif problem_type[-1] == 'BCDC_agree':
            if path_BCs[p] == max(path_BCs) and path_DCs[p] == max(path_DCs):
                path_choice_labels.append('maxBCDC')
            elif path_BCs[p] == max(path_BCs):
                path_choice_labels.append('maxBC')
            elif path_DCs[p] == max(path_DCs):
                path_choice_labels.append('maxDC')
            else:
                path_choice_labels.append('other')
        
        elif problem_type[-1] == 'BCDC_disagree':
            if path_BCs[p] == max(path_BCs):
                path_choice_labels.append('maxBC')
            elif path_DCs[p] == max(path_DCs):
                path_choice_labels.append('maxDC')
            else:
                path_choice_labels.append('other')

    problem_type, path_choice_labels = np.array(problem_type), np.array(path_choice_labels)
    poi = np.in1d(problem_type, ['BC_only','BCDC_agree','BCDC_disagree'])
    choices = path_choice_labels[poi]
    choices = choices[choices!='wrong']
    BCpref_score = np.in1d(choices, ['maxBCDC', 'maxBC']).mean() if len(choices) > 0 else None

    poi = np.in1d(problem_type, ['DC_only','BCDC_agree','BCDC_disagree'])
    choices = path_choice_labels[poi]
    choices = choices[choices!='wrong']
    DCpref_score = np.in1d(choices, ['maxBCDC', 'maxDC']).mean() if len(choices) > 0 else None
    
    if verbose:
        print(f'% predicted maxBC path among correctly-answered problems w diff avgBC scores: {BCpref_score}, baseline: {np.mean(BC_baseline)}')
        print(f'% predicted maxDC path among correctly-answered problems w diff avgDC scores: {DCpref_score}, baseline: {np.mean(DC_baseline)}')
        vals = ['maxBCDC', 'maxBC', 'maxDC', 'other', 'wrong']
        counts = [sum(path_choice_labels[problem_type=='BCDC_agree']==v) for v in vals]
        print(f'in problems where BC/DC agrees, {vals, counts}')
        vals = ['maxBC', 'maxDC', 'other', 'wrong']
        counts = [sum(path_choice_labels[problem_type=='BCDC_disagree']==v) for v in vals]
        print(f'in problems where BC/DC disagrees, {vals, counts}')
    
    # return problem_type, path_choice_labels
    return {'BCpref': BCpref_score, 'BC_baseline': np.mean(BC_baseline), 'DCpref': DCpref_score, 'DC_baseline': np.mean(DC_baseline)}

def path_choice_regression(batch, model, all_predictions, batch_metadata, path_score='max', response='binary', verbose=True):
    '''
    args
    ----
    path_score : str, 'max' or 'avg'
        indicates what path-level score to use as predictor
    response : str, 'binary' or 'prob'
        indicates whether to use binary path choice or path seq probability as response var
    '''
    
    def get_target_path_prob(model, mini_batch, target_paths):

        if type(model.transformer) == GraphAutoregTransformer:
            with torch.no_grad():
                out, _, _ = model.transformer.forward_rollout(mini_batch)
                result_dict = model.transformer.generate_result_dict(mini_batch, out)
            probs = torch.softmax(result_dict['pred'][0], dim=1)[:len(target_paths[0])]
            seq_probs = [
                 # product of probs of all nodes on path
                probs.gather(dim=1, index=torch.tensor(path).unsqueeze(1)).prod().item()
                for path in target_paths
            ]
        elif type(model.transformer) == GraphMaskedTransformer:
            with torch.no_grad():
                result_dict, _, _ = model.forward(mini_batch)
            probs = torch.softmax(result_dict['pred'][0], dim=1)[:len(target_paths[0])]
            seq_probs = [ 
                # product of probs of intermediate nodes
                probs.gather(dim=1, index=torch.tensor(path).unsqueeze(1))[1:-1].prod().item()
                for path in target_paths
            ]
        
        return seq_probs

    path_BC_scores, path_DC_scores = batch_metadata['path_BC_scores'], batch_metadata['path_DC_scores']

    # construct a table of path choices and path BC/DC scores
    choice, BC_scores, DC_scores = [], [], []

    for batch_idx, probe_idx in enumerate(batch['index'].tolist()):

        pred_path, target_paths, acc = all_predictions[probe_idx]
        if len(target_paths) == 1: continue # skip single-answer problems

        if response=='binary':
            if max(acc) != 1: continue # skip wrong problems
            choice.extend([int(i==np.argmax(acc)) for i in range(len(acc))])

        if response=='prob':
            mini_batch = {k:batch[k][batch_idx:batch_idx+1] for k in batch}
            seq_probs = get_target_path_prob(model, mini_batch, target_paths)
            choice.extend(seq_probs)

        BC_scores.extend(path_BC_scores[probe_idx][path_score])
        DC_scores.extend(path_DC_scores[probe_idx][path_score])
    
    # fit regression and get loss
    predictors = {'log_BC': np.log(BC_scores), 'log_DC': np.log(DC_scores)}
    losses = {'n_samples': len(choice)}
    if losses['n_samples'] == 0: # no correct multi-answer problems to train on
        return losses | {'log_BC': None, 'log_DC': None, 'bayes_factor (BC/DC)': None}

    data_likelihood = {}
    for predictor in predictors:
        x, y = predictors[predictor].reshape(-1, 1), choice
        if len(np.unique(x)) > 1:
            # standardize x only when there are multiple different instances
            x = (x - x.mean()) / x.std()
        if response=='binary':
            lr = sklearn.linear_model.LogisticRegression()
            lr.fit(x, y)
            loss = sklearn.metrics.log_loss(y_true=y, y_pred=lr.predict_proba(x))
            # compute bayes factor
            probs = torch.tensor(lr.predict_proba(x)).gather(dim=1, index=torch.tensor(y).unsqueeze(1))
            data_likelihood[predictor] = probs
        elif response=='prob':
            lr = sklearn.linear_model.LinearRegression()
            lr.fit(x, y)
            loss = sklearn.metrics.mean_squared_error(y_true=y, y_pred=lr.predict(x))
        losses[predictor] = loss
    if response=='binary':
        losses['bayes_factor (BC/DC)'] = (data_likelihood['log_BC'] / data_likelihood['log_DC']).prod().item()
    if verbose:
        print(losses)
    return losses

def temperature_modulated_softmax(logits, T=1.0):
    return F.softmax(logits / T, dim=-1)

def masked_graphembed_model_clamp_forward(model, batch, path_clamp_tensor):
    '''
    modified forward flow where we add clamp_tensor to the model default input embeddings
    clamp_tensor expected to have some content over <mask> tokens and 0 everywhere else
    for masked graph embed model, only clamp input_embed[:, 1:] assuming the first token is Graph context

    args
    ----
    clamp_tensor : torch.tensor, shape (bsz, max_path_len, embed_dim)

    return
    ------
    logits : torch.tensor, shape (bsz, max_path_len, out_dim)
    '''
    with torch.no_grad():
        x = model.input_embed(batch) # (bsz, 1+max_seq_len, embed_dim)
        x[:, -model.max_path_len:] = x[:, -model.max_path_len:] + path_clamp_tensor if path_clamp_tensor is not None else x[:, -model.max_path_len:]
        # x[:, 1:] = x[:, 1:] + path_clamp_tensor if path_clamp_tensor is not None else x[:, 1:] # hardcoded for graph context models
        attn_mask = model.create_attn_mask(x.shape[1], x.device)
        x, _, _ = model._encoder_forward(x, attn_mask=attn_mask, return_attn=False, return_reps=False)
        x = model.out(x) # (bsz, seq_len, out_dim)
        logits = x[:, -model.max_path_len:, :]
        return logits

def compute_intermediate_node_loss_acc(pred_logits, target_paths):
    '''
    loss/acc of one masked_problem
    TODO: need to check each candidate path of multi-answer problems. any way to parallelize?

    args
    ----
    pred_logits : torch.tensor, shape (max_path_len, out_dim)
    target_paths : list of list of int, target paths
    '''
    # compute min loss/max acc w.r.t. the candidate paths for this problem
    targets = [path[1:-1] for path in target_paths]
    losses = [F.cross_entropy(pred_logits[1:1+len(targets[0])], torch.tensor(nodes)) for nodes in targets]
    max_logit_nodes = pred_logits.argmax(-1) # (max_path_len,)
    accs = [(max_logit_nodes[1:1+len(targets[0])] == torch.tensor(nodes)).float().mean() for nodes in targets]
    return losses, accs