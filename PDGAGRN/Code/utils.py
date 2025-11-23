

class scRNADataset(Dataset):


    def __init__(self, train_set_df, num_gene):
        super(scRNADataset, self).__init__()

        self.train_pairs = train_set_df.iloc[:, :2].values.astype(np.int64)
        self.train_labels = train_set_df.iloc[:, -1].values.astype(np.float32)
        self.num_gene = num_gene

    def __getitem__(self, idx):

        return self.train_pairs[idx], self.train_labels[idx]

    def __len__(self):

        return len(self.train_labels)

    def Adj_Generate(self, loop=False):

        adj = sp.dok_matrix((self.num_gene, self.num_gene), dtype=np.float32)

        positive_pairs = self.train_pairs[self.train_labels == 1]

        for tf, target in positive_pairs:
            adj[tf, target] = 1.0
            adj[target, tf] = 1.0

        if loop:

            adj = adj + sp.identity(self.num_gene)


        return adj.tocoo()


class LoadData:


    def __init__(self, data, normalize=True):
        self.data = data
        self.normalize = normalize

    def _data_normalize(self, data_array):

        standard = StandardScaler()

        epr = standard.fit_transform(data_array.T)

        return epr.T

    def get_exp_data(self):

        data_feature = self.data.values
        if self.normalize:
            data_feature = self._data_normalize(data_feature)
        return data_feature.astype(np.float32)


def adj_coo_to_edge_index(adj):

    row = torch.from_numpy(adj.row.astype(np.int64))
    col = torch.from_numpy(adj.col.astype(np.int64))
    edge_index = torch.stack([row, col], dim=0)
    return edge_index


def compute_true_diffusion_matrix(adj_coo, beta=0.5, device='cpu'):

    print("正在计算真实扩散矩阵 (基于热核)...")
    A = adj_coo.toarray()


    deg = np.sum(A, axis=1)
    deg[deg == 0] = 1e-9
    D_inv_sqrt = np.diag(np.power(deg, -0.5))
    L_norm = np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt


    S = expm(-beta * L_norm)


    S_torch = torch.from_numpy(S).float()
    S_torch = F.relu(S_torch)
    row_sums = S_torch.sum(dim=1, keepdim=True)
    row_sums[row_sums < 1e-9] = 1.0
    S_normalized = S_torch / row_sums

    return S_normalized.to(device)


def Evaluation(y_true, y_pred):


    y_p_np = y_pred.cpu().detach().numpy().flatten()
    y_t_np = y_true.cpu().detach().numpy().flatten().astype(int)

    # 检查真实标签是否包含多个类别，否则无法计算AUC/AUPR
    if len(np.unique(y_t_np)) < 2:
        print("警告: 评估时真实标签只包含一个类别，AUC/AUPR未定义。")
        return np.nan, np.nan

    auc = roc_auc_score(y_true=y_t_np, y_score=y_p_np)
    aupr = average_precision_score(y_true=y_t_np, y_score=y_p_np)

    return auc, aupr