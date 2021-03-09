import dgl


def graph_from_arrays(lit_features, clause_features, adj_arrays):
    # print(np.shape(list(zip(adj_arrays["cols_arr"], adj_arrays["rows_arr"]))))
    G = dgl.heterograph(
                {('literal', 'l2c', 'clause') : list(zip(adj_arrays["cols_arr"], adj_arrays["rows_arr"])),
                 ('clause', 'c2l', 'literal') : list(zip(adj_arrays["rows_arr"], adj_arrays["cols_arr"]))},
                {'literal': len(lit_features),
                 'clause': adj_arrays['rows_arr'].max()+1})

    if lit_features is not None:
	    G.nodes['literal'].data['literal_feats'] = lit_features
    if clause_features is not None:
	    G.nodes['clause'].data['clause_ids']  = clause_features[:, 0]
	    G.nodes['clause'].data['clause_feats']  = clause_features[:, 1:-4]
   
    return G



def graph_from_adj(lit_features, clause_features, adj_matrix):
    ind = adj_matrix.coalesce().indices().t().tolist()
    ind_t = adj_matrix.t().coalesce().indices().t().tolist()

    G = dgl.heterograph(
                {('literal', 'l2c', 'clause') : ind_t,
                 ('clause', 'c2l', 'literal') : ind},
                {'literal': adj_matrix.shape[1],
                 'clause': adj_matrix.shape[0]})
    
    if lit_features is not None:
    	G.nodes['literal'].data['literal_feats'] = lit_features
    if clause_features is not None:
    	G.nodes['clause'].data['clause_feats'] = clause_features

    return G
