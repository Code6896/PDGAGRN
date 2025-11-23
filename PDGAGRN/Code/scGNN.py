import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class DynamicGraphLearner(nn.Module):


    def __init__(self, node_embedding_dim, hidden_dim=64, alpha_leaky_relu=0.2):
        super(DynamicGraphLearner, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_embedding_dim * 2, hidden_dim),
            nn.LeakyReLU(alpha_leaky_relu),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, node_embeddings, candidate_edge_index):

        if candidate_edge_index.shape[1] == 0:
            return torch.empty(0, device=node_embeddings.device)

        source_nodes = candidate_edge_index[0]
        target_nodes = candidate_edge_index[1]

        source_embs = node_embeddings[source_nodes]
        target_embs = node_embeddings[target_nodes]


        edge_pair_embs = torch.cat([source_embs, target_embs], dim=1)


        edge_logits = self.mlp(edge_pair_embs).squeeze(-1)

        return torch.sigmoid(edge_logits)



class GATEncoderForPretraining(nn.Module):


    def __init__(self, input_dim, hidden_dims, num_heads_list, alpha, dropout_rate=0.2):
        super(GATEncoderForPretraining, self).__init__()
        self.gat_layers = nn.ModuleList()
        self.alpha = alpha
        self.dropout_rate = dropout_rate

        current_channels = input_dim

        self.gat_layers.append(
            GATConv(current_channels, hidden_dims[0], heads=num_heads_list[0], concat=True, dropout=dropout_rate))
        current_channels = hidden_dims[0] * num_heads_list[0]

        self.gat_layers.append(
            GATConv(current_channels, hidden_dims[1], heads=num_heads_list[1], concat=True, dropout=dropout_rate))
        current_channels = hidden_dims[1] * num_heads_list[1]

        self.gat_layers.append(
            GATConv(current_channels, hidden_dims[2], heads=num_heads_list[2], concat=False, dropout=dropout_rate))

    def forward(self, x, edge_index):
        current_x = x
        for i, layer in enumerate(self.gat_layers):
            current_x = layer(current_x, edge_index)

            if i < len(self.gat_layers) - 1:
                current_x = F.elu(current_x)
                current_x = F.dropout(current_x, p=self.dropout_rate, training=self.training)
        return current_x


class DiffusionPretrainer(nn.Module):


    def __init__(self, gat_encoder, final_gat_emb_dim, num_nodes, decoder_hidden_dim=128):
        super(DiffusionPretrainer, self).__init__()
        self.encoder = gat_encoder
        self.decoder = nn.Sequential(
            nn.Linear(final_gat_emb_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(decoder_hidden_dim, num_nodes)
        )

    def forward(self, x, edge_index):
        node_embeddings = self.encoder(x, edge_index)
        predicted_logits = self.decoder(node_embeddings)

        return F.log_softmax(predicted_logits, dim=1)



class IntegratedGRNModel(nn.Module):


    def __init__(self, input_dim, hidden_dims, output_dim, num_heads, alpha, device,
                 decoder_type, num_nodes, num_node_types, dgl_hidden_dim=64):
        super(IntegratedGRNModel, self).__init__()
        self.device = device
        self.alpha = alpha
        self.decoder_type = decoder_type
        self.num_nodes = num_nodes
        self.num_node_types = num_node_types


        self.gat_layers = nn.ModuleList()
        current_channels = input_dim

        self.gat_layers.append(GATConv(current_channels, hidden_dims[0], heads=num_heads[0], concat=True, dropout=0.2))
        current_channels = hidden_dims[0] * num_heads[0]

        self.gat_layers.append(GATConv(current_channels, hidden_dims[1], heads=num_heads[1], concat=True, dropout=0.2))
        current_channels = hidden_dims[1] * num_heads[1]

        self.gat_layers.append(GATConv(current_channels, hidden_dims[2], heads=num_heads[2], concat=False, dropout=0.2))

        self.gat_layer_output_dims = [
            hidden_dims[0] * num_heads[0],
            hidden_dims[1] * num_heads[1],
            hidden_dims[2]
        ]
        self.final_gat_emb_dim = hidden_dims[2]


        dgl_input_dim = self.gat_layer_output_dims[0]
        self.dynamic_graph_learner = DynamicGraphLearner(dgl_input_dim, dgl_hidden_dim, alpha)


        self.hierarchical_fusion_weights = nn.Parameter(torch.ones(len(self.gat_layers)))
        self.fusion_projection_layers = nn.ModuleList([
            nn.Linear(dim_in, self.final_gat_emb_dim) for dim_in in self.gat_layer_output_dims
        ])


        if self.num_node_types <= 0:
            raise ValueError("提供有效的节点类型数量 (num_node_types > 0)")
        self.role_specific_transforms = nn.ModuleList([
            nn.Linear(self.final_gat_emb_dim, self.final_gat_emb_dim) for _ in range(num_node_types)
        ])


        self.tf_linear1 = nn.Linear(self.final_gat_emb_dim, output_dim)
        self.target_linear1 = nn.Linear(self.final_gat_emb_dim, output_dim)
        if self.decoder_type == 'MLP':
            self.decoder_mlp = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim // 2),
                nn.ReLU(),
                nn.Linear(output_dim // 2, 1)
            )

        self.reset_parameters()


    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu', self.alpha)
        for layer in self.gat_layers:
            layer.reset_parameters()
        for layer in self.fusion_projection_layers:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
        for layer in self.role_specific_transforms:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
        nn.init.xavier_uniform_(self.tf_linear1.weight, gain=gain)
        nn.init.xavier_uniform_(self.target_linear1.weight, gain=gain)
        if self.decoder_type == 'MLP':
            for layer in self.decoder_mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))

    def _encode_gat_and_collect_intermediates(self, x, edge_index):

        intermediate_outputs = []
        current_x = x
        for i, layer in enumerate(self.gat_layers):
            current_x = layer(current_x, edge_index)
            current_x = F.elu(current_x)
            current_x = F.dropout(current_x, p=0.2, training=self.training)
            intermediate_outputs.append(current_x)
        return intermediate_outputs

    def _apply_adaptive_fusion(self, intermediate_outputs, node_type_ids):


        normalized_weights = F.softmax(self.hierarchical_fusion_weights, dim=0)
        projected_outputs = [self.fusion_projection_layers[i](out) for i, out in enumerate(intermediate_outputs)]

        fused_embedding = torch.zeros_like(projected_outputs[-1])
        for i, proj_out in enumerate(projected_outputs):
            fused_embedding += normalized_weights[i] * proj_out

        fused_embedding = F.elu(fused_embedding)


        final_embedding = torch.zeros_like(fused_embedding)
        for type_id in range(self.num_node_types):
            mask = (node_type_ids == type_id)
            if mask.sum() > 0:
                final_embedding[mask] = self.role_specific_transforms[type_id](fused_embedding[mask])

        return F.elu(final_embedding)

    def decode(self, final_embeddings, sample_indices):

        tf_final_embs = F.elu(self.tf_linear1(final_embeddings))
        target_final_embs = F.elu(self.target_linear1(final_embeddings))

        tf_final_embs = F.dropout(tf_final_embs, p=0.1, training=self.training)
        target_final_embs = F.dropout(target_final_embs, p=0.1, training=self.training)

        train_tf_embs = tf_final_embs[sample_indices[:, 0]]
        train_target_embs = target_final_embs[sample_indices[:, 1]]

        if self.decoder_type == 'dot':
            pred = torch.sum(torch.mul(train_tf_embs, train_target_embs), dim=1).view(-1, 1)
        elif self.decoder_type == 'cosine':
            pred = F.cosine_similarity(train_tf_embs, train_target_embs, dim=1).view(-1, 1)
        elif self.decoder_type == 'MLP':
            pred = self.decoder_mlp(torch.cat([train_tf_embs, train_target_embs], dim=1))
        else:
            raise TypeError(f"不支持的解码器类型: '{self.decoder_type}'")
        return pred

    def forward(self, x_features, base_edge_index, downstream_sample_indices, node_type_ids,
                dgl_candidate_edge_index, dgl_edge_threshold=0.5):

        with torch.no_grad() if not self.training else torch.enable_grad():
            dgl_input_embs = F.elu(self.gat_layers[0](x_features, base_edge_index))


        learned_edge_weights = self.dynamic_graph_learner(dgl_input_embs, dgl_candidate_edge_index)

        dynamic_edges_mask = learned_edge_weights > dgl_edge_threshold
        effective_dynamic_edges = dgl_candidate_edge_index[:, dynamic_edges_mask]


        effective_edge_index = torch.cat([base_edge_index, effective_dynamic_edges], dim=1)
        effective_edge_index = torch.unique(effective_edge_index, dim=1)


        intermediate_gat_outputs = self._encode_gat_and_collect_intermediates(x_features, effective_edge_index)


        final_node_embeddings = self._apply_adaptive_fusion(intermediate_gat_outputs, node_type_ids)


        predicted_scores = self.decode(final_node_embeddings, downstream_sample_indices)

        return predicted_scores