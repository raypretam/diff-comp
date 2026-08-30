"""
Quick test: Check if model is actually making predictions or returning garbage
"""
import torch
from options import get_args
from models.ddim_bitdit import BitDit
from data.ner.necti_dataset import NeCTILabelSet
import sys

sys.argv = ['test', '--config_file', 'necti_finegrain_xlmr.yaml']
args = get_args()

print("="*80)
print("DEBUG: Checking Model Inference")
print("="*80)

# Load label set
label_set = NeCTILabelSet(
    data_path=args.data_path,
    granularity=args.granularity,
    use_context=args.use_context
)

# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BitDit(
    device=device,
    num_classes=len(label_set),
    backbone=args.backbone,
    time_steps=args.time_steps,
    sampling_steps=args.sampling_steps,
    noise_schedule=args.noise_schedule,
    ddim_sampling_eta=args.ddim_sampling_eta,
    self_condition=args.self_condition,
    snr_scale=args.snr_scale,
    dataset=f"necti_{args.granularity}",
    dim_model=args.dim_model,
    dim_time=args.dim_time,
    objective=args.objective,
    loss_type=args.loss_type,
    add_lstm=args.add_lstm,
    freeze_bert=args.freeze_bert,
    max_length=args.max_length,
    depth=args.depth,
    num_labels=len(label_set),
    compound_aware=getattr(args, 'compound_aware', False),
    compound_pooling=getattr(args, 'compound_pooling', 'mean'),
    use_graph_encoder=getattr(args, 'use_graph_encoder', False),
    num_gnn_layers=getattr(args, 'num_gnn_layers', 2),
    label_set=label_set if getattr(args, 'compound_aware', False) else None,
).to(device)

model.eval()

print(f"\nModel created. Compound-aware: {model.compound_aware}")
print(f"Use graph encoder: {model.use_graph_encoder if hasattr(model, 'use_graph_encoder') else 'N/A'}")

# Create dummy input
batch_size = 2
seq_len = 10
input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
attention_mask = torch.ones(batch_size, seq_len).to(device)
seq_labels = torch.randint(0, len(label_set), (batch_size, seq_len)).to(device)

print(f"\nInput shapes:")
print(f"  input_ids: {input_ids.shape}")
print(f"  attention_mask: {attention_mask.shape}")
print(f"  seq_labels: {seq_labels.shape}")

# Run inference
print(f"\n{'='*80}")
print("Running model inference...")
print(f"{'='*80}")

with torch.no_grad():
    try:
        predictions, path_x = model(input_ids, attention_mask, seq_labels, None)
        
        print(f"\nSuccess! Got predictions.")
        print(f"  predictions shape: {predictions.shape}")
        print(f"  predictions dtype: {predictions.dtype}")
        print(f"  predictions range: [{predictions.min().item()}, {predictions.max().item()}]")
        print(f"  Sample predictions[0]: {predictions[0].cpu().tolist()}")
        
        # Check if all predictions are the same (bad sign)
        unique_preds = torch.unique(predictions[0]).cpu().tolist()
        print(f"  Unique values in first prediction: {unique_preds[:20]}")
        
        if len(unique_preds) == 1:
            print(f"\n⚠️  WARNING: All predictions are the same value! Model may not be trained.")
        
        # Convert to label strings
        pred_labels = [label_set.id2label(int(p)) for p in predictions[0].cpu().tolist()]
        print(f"  Sample predicted labels: {pred_labels[:10]}")
        
        # Check if predictions are out of range
        if predictions.max() >= len(label_set) or predictions.min() < 0:
            print(f"\n⚠️  ERROR: Predictions out of range! Max label ID: {len(label_set)-1}")
        else:
            print(f"\n✓ Predictions are in valid range [0, {len(label_set)-1}]")
            
    except Exception as e:
        print(f"\n❌ ERROR during inference: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("If you see valid predictions above, the model is working.")
print("If all predictions are the same or errors occur, there's a problem with the model.")
print("="*80)
