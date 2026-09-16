import itertools
import logging
import numpy as np
import pandas as pd
import math
from reactome import ReactomeNetwork


def get_map_from_layer(layer_dict):
    pathways = list(layer_dict.keys())
    print ('pathways', len(pathways))
    genes = list(itertools.chain.from_iterable(layer_dict.values()))
    genes = list(np.unique(genes))
    print ('genes', len(genes))

    n_pathways = len(pathways)
    n_genes = len(genes)

    mat = np.zeros((n_pathways, n_genes))
    for p, gs in layer_dict.items():
        g_inds = [genes.index(g) for g in gs]
        p_ind = pathways.index(p)
        mat[p_ind, g_inds] = 1

    df = pd.DataFrame(mat, index=pathways, columns=genes)
    return df.T


def get_layer_maps(genes, n_levels=5, direction='root_to_leaf'):
    reactome_layers = ReactomeNetwork().get_layers(n_levels, direction)
    filtering_index = genes
    maps = []
    for i, layer in enumerate(reactome_layers[::-1]):
        print ('layer #', i)
        mapp = get_map_from_layer(layer)
        filter_df = pd.DataFrame(index=filtering_index)
        print ('filtered_map', filter_df.shape)
        filtered_map = filter_df.merge(mapp, right_index=True, left_index=True, how='left')
        print ('filtered_map', filter_df.shape)

        filtered_map = filtered_map.fillna(0)
        print ('filtered_map', filter_df.shape)
        filtering_index = filtered_map.columns
        logging.info('layer {} , # of edges  {}'.format(i, filtered_map.sum().sum()))
        maps.append(filtered_map)
    return maps


def rename_affyprobe_to_genename(l1k_data_df, l1k_features, map_source_address):
    """
    map input dataframe column name from affy prob id to gene names
    
    """
    meta = pd.read_csv(map_source_address)
    meta_gene_probID = meta.set_index('probe_id')
    d = dict(zip(meta_gene_probID.index, meta_gene_probID['symbol']))
    l1k_features_gn = [d[l] for l in l1k_features]
    l1k_data_df = l1k_data_df.rename(columns=d)

    return l1k_data_df, l1k_features_gn


def align_labels(name_list, value_list):
    """
    align numerical moa class labels to moa class labels
    
    """
    align_dict = {}
    for i in range(len(name_list)):
        align_dict[name_list[i]] = value_list[i]
    return align_dict
