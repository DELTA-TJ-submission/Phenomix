import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset, Sampler
import torch.nn.utils.prune as prune
from captum.attr import IntegratedGradients,LayerIntegratedGradients
import numpy as np
import pandas as pd

class PhenomixNet(nn.Module):
    def __init__(self, reactome_knowledge, gene_dim, cp_dim):
        super(PhenomixNet, self).__init__()
        self.gene_dim = gene_dim
        self.cp_dim = cp_dim
        # remove the pathways that do not link any gene
        self.reactome_knowledge = reactome_knowledge.loc[:, (reactome_knowledge == 1).any(axis=0)]
        self.generate_layers()
        self.mask_prune(layer_1_fc=False, layer_1_test=False)

    def generate_layers(self):
        self.ge_layer = nn.Linear(self.gene_dim, self.reactome_knowledge.shape[1])
        self.cp_layer = nn.Linear(self.cp_dim, 100)
        self.layer_2 = nn.Linear(self.reactome_knowledge.shape[1]+100, 1)

    def mask_prune(self, layer_1_fc=False, layer_1_test=False):
        """
        restrict the neural network links between genes and pathways according to the Reactome knowledge

        """
        mapp = torch.tensor(np.array(self.reactome_knowledge, dtype=np.float32)).T
        self.ge_layer = prune.custom_from_mask(
            self.ge_layer, name='weight', mask=mapp)

    def forward(self, x):
        ge_x = x[:, self.cp_dim:]
        cp_x = x[:, :self.cp_dim]
        ge_x = torch.relu(self.ge_layer(ge_x))
        cp_x = torch.relu(self.cp_layer(cp_x))
        x = torch.cat((ge_x,cp_x),dim=1)
        x = torch.sigmoid(self.layer_2(x))
        return x



class MOADataset(Dataset):
    def __init__(self, df):
        self.data_frame = df

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        train_x = self.data_frame.iloc[idx, :-1]
        train_x = torch.tensor(train_x, dtype=torch.float32)
        train_y = self.data_frame.iloc[idx, -1]
        train_y = torch.tensor(train_y, dtype=torch.float32)
        return train_x, train_y

class MoAResult():
    def __init__(self):
        self.CP_MoA_pattern = None
        self.CP_MoA_feature = None
        self.CP_MoA_feature_occurrence_count = None
        self.top10_genes = None
        self.top10_pathways = None

def moa_cp_explain_analysis(top10_cps):
    """
    The main idea of CP-MoA-Explain analysis is to conduct frequency statistics for Featuregroup, and Channel for top 10 most important renamed CP features and select the feature with highest frequency, 
    which indicates the most significant relationship to the drug MoA, as the MoA-CP feature, then cluster MoA-CP features via SpectralClustering31 and finally analyze the patterns of clustering results.
    """
    # rename top10 cp features and frequency statistics
    feature_dict = {}
    for top10_cp in top10_cps:
        cp_features = top10_cp.split('_')
        if cp_features[1]=='AreaShape':
            renamed_cp_feature = f'{cp_features[0]}_AreaShape'
        elif cp_features[1]=='Correlation':
            renamed_cp_feature = f'{cp_features[3]}_{cp_features[4]}_Correlation'
        else:
            renamed_cp_feature = f'{cp_features[3]}_{cp_features[1]}' 
        if renamed_cp_feature not in feature_dict:
            feature_dict[renamed_cp_feature] = 1
        else:
            feature_dict[renamed_cp_feature] += 1
    # sorted the renamed top10 features and select the feature with highest frequency as the MoA-CP feature
    CP_MoA_feature = sorted(feature_dict.items(),key=lambda x:x[1], reverse=True)[0][0]
    CP_MoA_feature_occurrence_count = sorted(feature_dict.items(),key=lambda x:x[1], reverse=True)[0][1]
    # link the MoA-CP feature with the CP pattern
    if 'Nuclei_AreaShape' in CP_MoA_feature or ('DNA' in CP_MoA_feature and 'Correlation' not in CP_MoA_feature):
        CP_MoA_pattern = 'Nuclei'
    elif 'Cytoplasm_AreaShape' in CP_MoA_feature or 'Cells_AreaShape' in CP_MoA_feature:
        CP_MoA_pattern = 'Cytoplasm/Cytomembrane AreaShape'
    elif ('ER' in CP_MoA_feature or 'AGP' in CP_MoA_feature or 'RNA' in CP_MoA_feature) and 'Correlation' not in CP_MoA_feature:
        CP_MoA_pattern = 'RNA/Protein'
    elif ('Mito' in CP_MoA_feature) and ('Correlation' not in CP_MoA_feature):
        CP_MoA_pattern = 'Mitochondrion'
    elif 'Correlation' in CP_MoA_feature:
        CP_MoA_pattern = 'Correlation'  
    return CP_MoA_pattern, CP_MoA_feature, CP_MoA_feature_occurrence_count

def cp_moa_discovery(explain_data, target_class, explain_model, target_class_explain_data):
    top10_cps_columns = ['10th_important_cp', '9th_important_cp', '8th_important_cp', '7th_important_cp', '6th_important_cp','5th_important_cp', '4th_important_cp', '3rd_important_cp', '2nd_important_cp', '1st_important_cp']
    gene_and_cp_ig = IntegratedGradients(explain_model)
    gene_and_cp_attribution = gene_and_cp_ig.attribute(target_class_explain_data).detach().numpy()
    cp_attribution = gene_and_cp_attribution[:,:explain_model.cp_dim]
    cp_attribution_top10 = [np.argsort(i)[-10:] for i in cp_attribution]
    explain_data.loc[explain_data['Metadata_moa_num'] ==  target_class, top10_cps_columns] = cp_attribution_top10
    top10_cps = np.unique(cp_attribution_top10, return_counts=True)
    top10_cps = top10_cps[0][np.argsort(top10_cps[1])][-10:]
    top10_cps = explain_data.columns[top10_cps].tolist()
    CP_MoA_pattern, CP_MoA_feature, CP_MoA_feature_occurrence_count = moa_cp_explain_analysis(top10_cps)
    return CP_MoA_pattern, CP_MoA_feature, CP_MoA_feature_occurrence_count, explain_data

def gene_and_pathway_moa_discovery(explain_data, target_class, explain_model, target_class_explain_data):
    top10_pathways_columns = ['10th_important_pathway', '9th_important_pathway', '8th_important_pathway', '7th_important_pathway', '6th_important_pathway','5th_important_pathway', '4th_important_pathway','3rd_important_pathway', '2nd_important_pathway', '1st_important_pathway']
    top10_genes_columns = ['10th_important_gene', '9th_important_gene', '8th_important_gene', '7th_important_gene', '6th_important_gene','5th_important_gene', '4th_important_gene','3rd_important_gene', '2nd_important_gene', '1st_important_gene']
    gene_and_cp_ig = IntegratedGradients(explain_model)
    pathway_ig = LayerIntegratedGradients(explain_model, explain_model.ge_layer)
    gene_and_cp_attribution = gene_and_cp_ig.attribute(target_class_explain_data).detach().numpy()
    gene_attribution = gene_and_cp_attribution[:,explain_model.cp_dim:]
    #pathway
    pathway_attribution = pathway_ig.attribute(
        target_class_explain_data).detach().numpy()
    pathway_attribution_top10 = [np.argsort(
        i)[-10:] for i in pathway_attribution]
    explain_data.loc[explain_data['Metadata_moa_num'] ==
                    target_class, top10_pathways_columns] = pathway_attribution_top10
    top10_pathways = np.unique(
        pathway_attribution_top10, return_counts=True)
    top10_pathways = top10_pathways[0][np.argsort(top10_pathways[1])][-10:]
    top10_pathways = explain_model.reactome_knowledge.columns[top10_pathways]

    #gene
    gene_attribution_top10 = [np.argsort(i)[-10:] for i in gene_attribution]
    explain_data.loc[explain_data['Metadata_moa_num'] ==
                    target_class, top10_genes_columns] = gene_attribution_top10
    top10_genes = np.unique(
        gene_attribution_top10, return_counts=True)
    top10_genes = top10_genes[0][np.argsort(top10_genes[1])][-10:]
    top10_genes = explain_model.reactome_knowledge.index[top10_genes].tolist()
    return top10_genes, top10_pathways, explain_data

def moa_discovery(explain_data, target_class, explain_model, align_dict, reactome_pathway, EXPLAIN_FOR_INDIVIDUAL_DRUG=True):
    important_feature_columns = pd.DataFrame(np.zeros((explain_data.shape[0], 30)),
                                            index=explain_data.index, columns=['10th_important_cp', '9th_important_cp', '8th_important_cp', '7th_important_cp', '6th_important_cp','5th_important_cp', '4th_important_cp', '3rd_important_cp', '2nd_important_cp', '1st_important_cp',
                                            '10th_important_gene', '9th_important_gene', '8th_important_gene', '7th_important_gene', '6th_important_gene','5th_important_gene', '4th_important_gene','3rd_important_gene', '2nd_important_gene', '1st_important_gene',
                                                                                '10th_important_pathway', '9th_important_pathway', '8th_important_pathway', '7th_important_pathway', '6th_important_pathway','5th_important_pathway', '4th_important_pathway', '3rd_important_pathway', '2nd_important_pathway', '1st_important_pathway'],dtype='int32')
    explain_data = pd.concat((explain_data, important_feature_columns), axis=1)
    target_class_explain_data = explain_data.loc[explain_data['Metadata_moa_num']
                                == target_class].iloc[:, :-32]
    target_class_explain_data = torch.tensor(
        np.array(target_class_explain_data, dtype=np.float64), dtype=torch.float32)
    CP_MoA_pattern, CP_MoA_feature, CP_MoA_feature_occurrence_count, explain_data = cp_moa_discovery(explain_data, target_class, explain_model, target_class_explain_data)
    top10_genes, top10_pathways, explain_data = gene_and_pathway_moa_discovery(explain_data, target_class, explain_model, target_class_explain_data)
    top10_pathways = [reactome_pathway.loc[reactome_pathway['reactome_id'] == i]['pathway_name'].values[0] for i in top10_pathways]
    moa_discovery_result = {}
    moa_result = MoAResult()
    moa_result.CP_MoA_pattern, moa_result.CP_MoA_feature, moa_result.CP_MoA_feature_occurrence_count, moa_result.top10_genes, moa_result.top10_pathways= CP_MoA_pattern, CP_MoA_feature, CP_MoA_feature_occurrence_count, top10_genes, top10_pathways
    moa_discovery_result[align_dict[target_class]] = moa_result
    
    if EXPLAIN_FOR_INDIVIDUAL_DRUG:
        explain_data.iloc[:, -30:] = explain_data.iloc[:, -30:].astype('int32')
        # find all drugs in the MoA class
        compounds = explain_data.loc[explain_data['Metadata_moa_num'] == target_class]['Compounds'].unique().tolist()
        # moa discovery for individual drugs
        for compound in compounds:
            top10_genes = np.unique(
                explain_data.loc[explain_data['Compounds'] == compound].iloc[:, -20:-10], return_counts=True)
            top10_genes = top10_genes[0][np.argsort(
                top10_genes[1])][-10:]
            top10_genes = explain_model.reactome_knowledge.index[top10_genes].tolist()
            top10_pathways = np.unique(
                explain_data.loc[explain_data['Compounds'] == compound].iloc[:, -10:], return_counts=True)
            top10_pathways = top10_pathways[0][np.argsort(top10_pathways[1])][-10:]
            top10_pathways = explain_model.reactome_knowledge.columns[top10_pathways]
            top10_pathways = [reactome_pathway.loc[reactome_pathway['reactome_id'] ==
                                    i]['pathway_name'].values[0] for i in top10_pathways]
            top10_cps = np.unique(
                explain_data.loc[explain_data['Compounds'] == compound].iloc[:, -30:-20], return_counts=True)
            top10_cps = top10_cps[0][np.argsort(top10_cps[1])][-10:]
            top10_cps = explain_data.columns[top10_cps].tolist()
            CP_MoA_pattern, CP_MoA_feature, CP_MoA_feature_occurrence_count = moa_cp_explain_analysis(top10_cps)
            moa_result = MoAResult()
            moa_result.CP_MoA_pattern, moa_result.CP_MoA_feature, moa_result.CP_MoA_feature_occurrence_count, moa_result.top10_genes, moa_result.top10_pathways= CP_MoA_pattern, CP_MoA_feature, CP_MoA_feature_occurrence_count, top10_genes, top10_pathways
            moa_discovery_result[compound] = moa_result
    return moa_discovery_result
