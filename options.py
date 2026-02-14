"""
@Date  : 2022/12/18
@Time  : 15:18
@Author: Ziyang Huang
@Email : huangzy0312@gmail.com
"""
import argparse
import yaml
import os


def get_parser():
    parser = argparse.ArgumentParser()

    # model args
    parser.add_argument("--base", default='ner')
    parser.add_argument("--config_file", default="resume.bert-base-uncased.ner.yaml", type=str)
    parser.add_argument("--dataset", default="fewnerd-few_nerd", type=str)
    parser.add_argument("--num_classes", default=3, type=int)
    parser.add_argument("--backbone", default='../plm/bert-base-chinese', type=str)
    parser.add_argument("--time_steps", default=1000, type=int)
    parser.add_argument("--sampling_steps", default=10, type=int)
    parser.add_argument("--ddim_sampling_eta", default=1., type=float)
    parser.add_argument("--self_condition", default=False, type=bool)
    parser.add_argument("--snr_scale", default=2., type=float)
    parser.add_argument("--dim_model", default=768, type=int)
    parser.add_argument("--encoder_depth", default=3, type=int, dest="the depth of tranformer encoder")
    parser.add_argument("--decoder_depth", default=6, type=int, dest="the depth of tranformer decoder")
    parser.add_argument("--dim_time", default=256, type=int)
    parser.add_argument("--objective", default='pred_x0', type=str)
    parser.add_argument("--noise_schedule", default="linear", type=str)
    parser.add_argument("--loss_type", default='l2', choices=['l1', 'l2'])
    parser.add_argument("--add_lstm", default=False, type=bool)
    parser.add_argument("--freeze_bert", default=False, type=bool)
    parser.add_argument("--decode_mode", default='bmes', type=str)
    parser.add_argument("--patch_size", default=4, type=int)
    parser.add_argument("--max_length", default=256, type=int)
    parser.add_argument("--network_architecture", default="transformer", type=str)
    parser.add_argument("--ensemble", default=False, type=bool)

    # training args
    parser.add_argument("--logger", default='None', type=str)
    parser.add_argument("--output_dir", default='output', type=str)
    parser.add_argument("--model_path", default='model.pt', type=str)
    parser.add_argument("--use_gpu", default=False, type=bool)
    parser.add_argument("--gpus", default=1)
    parser.add_argument("--max_steps", default=250000, type=int)
    parser.add_argument("--max_epochs", default=15, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=6, type=int, dest="num_workers for dataloader, 0 for debugging")
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--warmup_ratio", default=0.01, type=float)
    parser.add_argument("--optimizer_type", default='AdamW', type=str)
    parser.add_argument("--lr_scheduler_type", default='linear', type=str)
    parser.add_argument("--num_cycles", default=1, type=int)
    parser.add_argument("--lr_bert", default=5e-5, type=float)
    parser.add_argument("--lr_other", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--accumulation_steps", default=4, type=int)
    parser.add_argument("--test_path", action='store_true')
    parser.add_argument('--save_limit', type=int, default=3, help='Maximum number of checkpoints to keep (excluding best model)')
    
    # NeCTI-specific args
    parser.add_argument("--data_path", default="/home/pretam-pg/DepNeCTI/data/NeCTIS Model Data", type=str,
                       help="Base path to NeCTIS Model Data")
    parser.add_argument("--granularity", default="Coarse", type=str, choices=['Coarse', 'Finegrain', 'coarse', 'fine'],
                       help="Compound granularity level")
    parser.add_argument("--use_context", action='store_true',
                       help="Use 'With Context' data instead of 'Without Context' data")
    parser.add_argument("--depth", default=6, type=int, help="Depth of DiT model")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm for clipping")
    parser.add_argument("--use_cle_decoding", action='store_true', default=True,
                       help="Use Chu-Liu-Edmonds algorithm for structured decoding during inference")
    parser.add_argument("--patience", default=5, type=int, help="Early stopping patience (number of epochs)")
    parser.add_argument("--min_delta", default=0.0001, type=float, help="Minimum improvement for early stopping")
    
    # Compound-aware diffusion args
    parser.add_argument("--compound_aware", default=False, type=bool, 
                       help="Enable compound-level diffusion")
    parser.add_argument("--compound_pooling", default='mean', type=str, choices=['mean', 'max', 'attention', 'lstm'],
                       help="Pooling method for compound representations")
    parser.add_argument("--use_graph_encoder", default=False, type=bool,
                       help="Use Graph-Aware Encoder to model dependencies between compounds")
    parser.add_argument("--num_gnn_layers", default=2, type=int,
                       help="Number of GNN layers for inter-compound message passing")
    
    # Contrastive learning args
    parser.add_argument("--use_contrastive", default=False, type=bool,
                       help="Enable contrastive learning")
    parser.add_argument("--contrastive_weight", default=0.1, type=float,
                       help="Weight for contrastive loss")
    parser.add_argument("--contrastive_temp", default=0.07, type=float,
                       help="Temperature for InfoNCE contrastive loss")
    parser.add_argument("--contrastive_type", default='simple', type=str, choices=['simple', 'hierarchical'],
                       help="Type of contrastive learning")
    
    # SaCTI-specific args
    parser.add_argument("--language", default='auto', type=str,
                       help="Language for SaCTI: 'sacti', 'marathi', 'english', or 'auto'")
    
    args = parser.parse_args()
    default_path = os.path.join(os.getcwd(), "configs", args.config_file)
    with open(default_path, 'r') as f:
        default_args_from_file = yaml.load(f, Loader=yaml.FullLoader)
    parser.set_defaults(**default_args_from_file)

    return parser


def get_args():
    """Wrapper to get parsed arguments"""
    parser = get_parser()
    return parser.parse_args()