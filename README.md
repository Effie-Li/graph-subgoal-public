# graph-subgoal-public

This repo hosts the modeling and analysis code for:

Li, Y., & McClelland, J.L. (2025). Learning to decompose: Human-like subgoal preferences emerge in transformers learning graph traversal. *Submitted*.

## Abstract

Cognitive scientists have discovered normative and heuristic principles that capture
human subgoal preferences when partitioning problems into smaller ones. However, it
remains unclear where such preferences come from. In this work, we study the processes
through which these subgoal and partition preferences may arise over learning. We build
on the graph-based environments from prior work and use neural networks as model
learners to test if learning shortest-path graph traversal can lead to human-like path
decomposition. We find that simple transformer models develop a preference for paths
containing nodes that occur frequently on the shortest paths in the graph, consistent
with human subgoal preferences found in prior work. This preference is strongest early
in model learning, a phenomenon that might also be observed in human learners. We
also explore more explicit subgoal choices in models that learn shortest-path prediction
through a dynamic and iterative path completion process. These results complement
existing theoretical accounts of human task decomposition in graph-like environments,
and lay the ground for using neural networks and a learning perspective to study the
data distribution responsible for the emergence of human-like subgoal preferences.

If you use this work, please cite:
```
@article{LiMcClelland2025Learning,
  title={Learning to decompose: Human-like subgoal preferences emerge in transformers learning graph traversal},
  author={Li, Yuxuan and McClelland, James L},
  journal={Submitted},
  year={2025}
}
```
