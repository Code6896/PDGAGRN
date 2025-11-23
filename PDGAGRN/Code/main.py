import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
import yaml
# import argparse
import numpy as np
import pandas as pd
import random
import os

from scGNN import GATEncoderForPretraining, DiffusionPretrainer, PDGAGRN
from utils import scRNADataset, LoadData, adj_coo_to_edge_index, compute_true_diffusion_matrix, Evaluation
from torch_geometric.nn import knn_graph
from PytorchTools import EarlyStopping
from types import SimpleNamespace

def main():

    parser = argparse.ArgumentParser(description="集成GNN模型用于基因调控网络推断")

    parser.add_argument('--net_type', type=str, default='Specific', help="网络类型 (e.g., STRING, Specific)")
    parser.add_argument('--data_type', type=str, default='hESC', help="细胞类型 (e.g., hESC, mDC)")

    parser.add_argument('--num', type=int, default=500, help="网络规模 (e.g., 500, 1000)")
    parser.add_argument('--seed', type=int, default=8, help="全局随机种子")


    parser.add_argument('--hidden_dim', type=int, nargs=3, default=[256, 128, 64], help="GAT三层隐藏维度")

    parser.add_argument('--output_dim', type=int, default=1024, help="解码器前最终嵌入维度")
    parser.add_argument('--num_head', type=int, nargs=3, default=[3, 3, 6], help="GAT三层注意力头数")
    parser.add_argument('--alpha', type=float, default=0.2, help="LeakyReLU激活函数的alpha值")
    parser.add_argument('--decoder_type', type=str, default='MLP', choices=['dot', 'cosine', 'MLP'], help="解码器计算得分方式")
    parser.add_argument('--loop', type=bool, default=False, help="是否在邻接矩阵中添加自环")


    parser.add_argument('--batch_size', type=int, default=128, help="下游任务训练的批处理大小")
    parser.add_argument('--epochs_pretrain', type=int, default=200, help="图扩散预训练的周期数")
    parser.add_argument('--lr_pretrain', type=float, default=1e-6, help="预训练阶段的学习率")
    parser.add_argument('--epochs_finetune', type=int, default=200, help="下游任务微调的最大周期数")
    parser.add_argument('--lr_finetune', type=float, default=9e-6, help="微调阶段的学习率")
    parser.add_argument('--early_stopping_patience', type=int, default=15, help="早停机制的耐心值")


    parser.add_argument('--diffusion_beta', type=float, default=0.5, help="热核扩散矩阵的beta系数")
    parser.add_argument('--dgl_threshold', type=float, default=0.7, help="动态图学习中边的选择阈值")

    args = parser.parse_args()
    print("当前配置参数:", args)


    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"当前使用的设备: {device}")


    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dataset_base_path = os.path.join(project_root, 'Dataset', 'Benchmark Dataset')
    results_base_path = os.path.join(project_root, 'Result')

    dataset_path = os.path.join(dataset_base_path, f'{args.net_type} Dataset', args.data_type, f'TFs+{args.num}')
    exp_file = os.path.join(dataset_path, 'BL--ExpressionData.csv')
    tf_file = os.path.join(dataset_path, 'TF.csv')

    downstream_path = os.path.join(project_root, args.net_type, f'{args.data_type} {args.num}')
    train_file = os.path.join(downstream_path, 'Train_set.csv')
    val_file = os.path.join(downstream_path, 'Validation_set.csv')
    test_file = os.path.join(downstream_path, 'Test_set.csv')

    result_dir = os.path.join(results_base_path, args.net_type, f'{args.data_type} {args.num}')
    os.makedirs(result_dir, exist_ok=True)
    pretrained_encoder_path = os.path.join(result_dir, f'pretrained_encoder_seed{args.seed}.pt')
    finetuned_model_filename = f'finetuned_best_model_seed{args.seed}.pt'


    expression_df = pd.read_csv(exp_file, index_col=0)
    num_nodes = expression_df.shape[0]

    loader = LoadData(expression_df, normalize=True)
    features_np = loader.get_exp_data()
    features = torch.from_numpy(features_np).to(device)

    train_df = pd.read_csv(train_file, index_col=0)
    val_df = pd.read_csv(val_file, index_col=0)
    test_df = pd.read_csv(test_file, index_col=0)


    train_dataset_for_adj = scRNADataset(train_df, num_nodes)
    base_adj_coo = train_dataset_for_adj.Adj_Generate(loop=args.loop)
    base_edge_index = adj_coo_to_edge_index(base_adj_coo).to(device)




    k_for_knn = 10
    knn_edge_index = knn_graph(features, k=k_for_knn, loop=False, batch=None).to(device)



    dgl_candidate_edge_index = torch.cat([base_edge_index, knn_edge_index], dim=1)


    dgl_candidate_edge_index = torch.unique(dgl_candidate_edge_index, dim=1)



    tf_indices = pd.read_csv(tf_file, index_col=0)['index'].values
    node_type_ids = torch.ones(num_nodes, dtype=torch.long)
    node_type_ids[tf_indices] = 0  #
    node_type_ids = node_type_ids.to(device)
    num_node_types = len(torch.unique(node_type_ids))


    print("\n--- Phase 1: 图扩散预训练 ---")
    gat_encoder_pretrain = GATEncoderForPretraining(
        input_dim=features.shape[1], hidden_dims=args.hidden_dim,
        num_heads_list=args.num_head, alpha=args.alpha
    ).to(device)

    diffusion_pretrainer = DiffusionPretrainer(
        gat_encoder=gat_encoder_pretrain,
        final_gat_emb_dim=args.hidden_dim[2],
        num_nodes=num_nodes
    ).to(device)

    true_S_matrix = compute_true_diffusion_matrix(base_adj_coo, beta=args.diffusion_beta, device=device)
    optimizer_pretrain = Adam(diffusion_pretrainer.parameters(), lr=args.lr_pretrain)

    for epoch in range(args.epochs_pretrain):
        diffusion_pretrainer.train()
        optimizer_pretrain.zero_grad()
        predicted_log_probs = diffusion_pretrainer(features, base_edge_index)
        loss = F.kl_div(predicted_log_probs, true_S_matrix, reduction='batchmean')
        loss.backward()
        optimizer_pretrain.step()
        if (epoch + 1) % 10 == 0 or epoch == args.epochs_pretrain - 1:
            print(f"Pretrain Epoch {epoch + 1}/{args.epochs_pretrain}, KL Loss: {loss.item():.6f}")


    torch.save(gat_encoder_pretrain.gat_layers.state_dict(), pretrained_encoder_path)
    print(f"预训练完成，GAT编码器权重已保存到: {pretrained_encoder_path}")


    print("\n--- Phase 2: 下游任务微调 ---")

    model = IntegratedGRNModel(
        input_dim=features.shape[1], hidden_dims=args.hidden_dim, output_dim=args.output_dim,
        num_heads=args.num_head, alpha=args.alpha, device=device, decoder_type=args.decoder_type,
        num_nodes=num_nodes, num_node_types=num_node_types
    ).to(device)


    try:
        model.gat_layers.load_state_dict(torch.load(pretrained_encoder_path, map_location=device))
        print("成功加载预训练GAT权重。")
    except Exception as e:
        print(f"警告: 加载预训练GAT权重失败: {e}。将从随机初始化开始微调。")

    train_dataset = scRNADataset(train_df, num_nodes)
    val_pairs = torch.from_numpy(val_df.iloc[:, :2].values.astype(np.int64)).to(device)
    val_labels = torch.from_numpy(val_df.iloc[:, -1].values.astype(np.float32)).to(device)
    test_pairs = torch.from_numpy(test_df.iloc[:, :2].values.astype(np.int64)).to(device)
    test_labels = torch.from_numpy(test_df.iloc[:, -1].values.astype(np.float32)).to(device)


    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=g
    )

    optimizer = Adam(model.parameters(), lr=args.lr_finetune)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.9)
    early_stopping = EarlyStopping(save_dir=result_dir, patience=args.early_stopping_patience, verbose=True,
                                   save_model_name=finetuned_model_filename)

    for epoch in range(args.epochs_finetune):
        model.train()
        total_loss = 0
        for train_pairs, train_labels in train_loader:
            train_pairs, train_labels = train_pairs.to(device), train_labels.to(device).view(-1, 1)

            optimizer.zero_grad()
            pred_scores = model(
                x_features=features, base_edge_index=base_edge_index,
                downstream_sample_indices=train_pairs, node_type_ids=node_type_ids,
                dgl_candidate_edge_index=dgl_candidate_edge_index, dgl_edge_threshold=args.dgl_threshold
            )
            loss = F.binary_cross_entropy_with_logits(pred_scores, train_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0


        model.eval()
        with torch.no_grad():
            val_scores = model(
                x_features=features, base_edge_index=base_edge_index,
                downstream_sample_indices=val_pairs, node_type_ids=node_type_ids,
                dgl_candidate_edge_index=dgl_candidate_edge_index, dgl_edge_threshold=args.dgl_threshold
            )
            val_probs = torch.sigmoid(val_scores)
            val_auc, val_aupr = Evaluation(val_labels, val_probs)

        print(
            f"Finetune Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val AUC: {val_auc:.4f}, Val AUPR: {val_aupr:.4f}")

        scheduler.step()
        early_stopping(val_auc, model)
        if early_stopping.early_stop:
            print("早停机制触发，结束训练。")
            break


    print("\n--- 开始最终测试 ---")
    best_model_path = os.path.join(result_dir, finetuned_model_filename)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    with torch.no_grad():
        test_scores = model(
            x_features=features, base_edge_index=base_edge_index,
            downstream_sample_indices=test_pairs, node_type_ids=node_type_ids,
            dgl_candidate_edge_index=dgl_candidate_edge_index, dgl_edge_threshold=args.dgl_threshold
        )
        test_probs = torch.sigmoid(test_scores)
        test_auc, test_aupr = Evaluation(test_labels, test_probs)

    print(f"--- 最终测试结果 (数据集: {args.data_type}, 网络: {args.net_type}, 规模: {args.num}) ---")
    print(f"Test AUC: {test_auc:.6f}, AUPR: {test_aupr:.6f}")


if __name__ == '__main__':
    main()