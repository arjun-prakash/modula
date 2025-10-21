#!/usr/bin/env python3
"""
export_predictions_for_html.py - Export trained PINN weights to JavaScript format

This script converts the trained DeepONet weights to a format that can be
used in the HTML visualization.
"""

import numpy as np
import json
from deeponet import DeepONet


def export_weights_to_json(network, filepath='deeponet_weights.json'):
    """
    Export all network weights to JSON format for JavaScript
    
    Args:
        network: trained DeepONet
        filepath: output JSON file path
    """
    weights = {
        'W_branch1': network.W_branch1.tolist(),
        'b_branch1': network.b_branch1.tolist(),
        'W_branch2': network.W_branch2.tolist(),
        'b_branch2': network.b_branch2.tolist(),
        'W_trunk1': network.W_trunk1.tolist(),
        'b_trunk1': network.b_trunk1.tolist(),
        'W_trunk2': network.W_trunk2.tolist(),
        'b_trunk2': network.b_trunk2.tolist(),
        'W_out': network.W_out.tolist(),
        'b_out': network.b_out.tolist()
    }
    
    with open(filepath, 'w') as f:
        json.dump(weights, f)
    
    print(f"✓ Weights exported to {filepath}")
    print(f"  File size: {len(json.dumps(weights))/1024:.1f} KB")


def export_prediction_table(network, filepath='prediction_table.json', 
                            n_samples=1000):
    """
    Generate a lookup table of predictions for common scenarios
    
    This pre-computes predictions for various drone positions and gust conditions,
    allowing fast lookup in JavaScript without running the full network.
    
    Args:
        network: trained DeepONet
        filepath: output JSON file path
        n_samples: number of prediction samples to generate
    """
    print(f"Generating {n_samples} prediction samples...")
    
    predictions = []
    
    for i in range(n_samples):
        # Random drone position
        pos = np.random.randn(3) * 5.0
        vel = np.random.randn(3) * 2.0
        
        # Random gust features (simplified: 2 bursts)
        gust_features = np.random.randn(10) * 0.5
        
        # Random trunk (state + time + base wind)
        trunk = np.concatenate([pos, vel, [np.random.rand()], np.random.randn(3) * 0.5])
        
        # Get prediction
        pred = network.forward(
            gust_features.reshape(1, -1),
            trunk.reshape(1, -1)
        )[0]
        
        predictions.append({
            'pos': pos.tolist(),
            'vel': vel.tolist(),
            'gust_features': gust_features.tolist(),
            'prediction': pred.tolist()
        })
        
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{n_samples}")
    
    with open(filepath, 'w') as f:
        json.dump(predictions, f)
    
    print(f"✓ Prediction table exported to {filepath}")
    print(f"  File size: {len(json.dumps(predictions))/1024:.1f} KB")


if __name__ == "__main__":
    import sys
    
    # Load trained network
    network = DeepONet()
    
    try:
        network.load_weights("deeponet_weights.npz")
        print("✓ Loaded trained weights from deeponet_weights.npz\n")
    except:
        print("✗ Could not load trained weights!")
        print("Please train the network first:")
        print("  python training.py")
        print("  OR")
        print("  python main.py --train")
        sys.exit(1)
    
    # Export options
    print("="*70)
    print("EXPORT OPTIONS")
    print("="*70)
    print("1. Export full weights (for JavaScript implementation)")
    print("2. Export prediction lookup table (faster, approximate)")
    print("3. Export both")
    print("="*70)
    
    choice = input("Choose option (1/2/3): ").strip()
    
    if choice in ['1', '3']:
        export_weights_to_json(network, 'deeponet_weights.json')
    
    if choice in ['2', '3']:
        n_samples = input("Number of samples for lookup table (default 1000): ").strip()
        n_samples = int(n_samples) if n_samples else 1000
        export_prediction_table(network, 'prediction_table.json', n_samples)
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("To use in HTML visualization:")
    print("1. Include the exported JSON file(s) in your HTML")
    print("2. Implement JavaScript DeepONet forward pass")
    print("3. Load weights and use for predictions")
    print("\nSee: create_html_with_real_pinn.py for full implementation")
    print("="*70)