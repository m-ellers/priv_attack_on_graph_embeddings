import sklearn.naive_bayes as nb
import pipeline

import entities.parameters as p
import entities.embeddings.gem_embedding as gem
import entities.graphs.graph as g
import memory_access.memory_access as ma

if __name__ == '__main__':
    # --- attacked graph ---
    graph = g.Graph.init_karate_club_graph()

    # --- sample attacked nodes from graph and nodes used for training ----
    num_attacked_nodes_per_deg_level = 1  # num attacked nodes = 3 * num_attacked_nodes_per_deg_level (for 3 deg levels)
    num_training_graphs = 2

    sampled_attacked_and_training_nodes = pipeline.sample_nodes(
        graph=graph,
        num_attacked_nodes_pre_deg_level=num_attacked_nodes_per_deg_level,
        num_training_nodes=num_training_graphs)

    # --- parameters for experiment ---
    params = p.Parameters(graph=graph,
                          embedding=gem.GEMEmbedding.init_hope(dim=8),
                          num_of_bins=10,
                          classifier=nb.GaussianNB(),
                          dict_attacked_nodes_training_nodes=sampled_attacked_and_training_nodes)
    mem_acc = ma.MemoryAccess()  # file access class

    # --- run experiment ---
    pipeline.run(parameters=params, memory_access=mem_acc)
