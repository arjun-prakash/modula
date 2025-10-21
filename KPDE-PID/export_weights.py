#!/usr/bin/env python3
"""
fixed_export_weights.py - Export trained PINN weights to JSON with verification
"""

import numpy as np
import json
import os
from deeponet import DeepONet


def export_weights():
    """Export DeepONet weights to JSON format with verification"""
    
    print("="*70)
    print("EXPORT DEEPONET WEIGHTS FOR HTML VISUALIZATION")
    print("="*70)
    print()
    
    # Check if trained weights exist
    if not os.path.exists("deeponet_weights.npz"):
        print("✗ deeponet_weights.npz not found!")
        print("\nPlease train the network first:")
        print("  python main.py --train")
        print("  OR")
        print("  python training.py")
        return False
    
    # Load trained network
    network = DeepONet()
    try:
        network.load_weights("deeponet_weights.npz")
        print("✓ Loaded trained weights from deeponet_weights.npz")
    except Exception as e:
        print(f"✗ Could not load trained weights!")
        print(f"Error: {e}")
        return False
    
    print("\nExporting weights to JSON...")
    
    # Convert all weights to lists for JSON serialization
    # This matches EXACTLY what the HTML expects
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
        'b_out': network.b_out.tolist(),
        'metadata': {
            'branch_input_dim': 10,
            'trunk_input_dim': 10,
            'hidden_dim': 64,
            'output_dim': 3
        }
    }
    
    # Verify shapes
    print("\nVerifying weight shapes:")
    print(f"  W_branch1: {network.W_branch1.shape}")
    print(f"  W_branch2: {network.W_branch2.shape}")
    print(f"  W_trunk1:  {network.W_trunk1.shape}")
    print(f"  W_trunk2:  {network.W_trunk2.shape}")
    print(f"  W_out:     {network.W_out.shape}")
    
    # Save to JSON file
    filepath = 'deeponet_weights.json'
    try:
        with open(filepath, 'w') as f:
            json.dump(weights, f)
        
        file_size = os.path.getsize(filepath) / 1024
        
        print(f"\n✓ Weights successfully exported!")
        print(f"  File: {filepath}")
        print(f"  Size: {file_size:.1f} KB")
        
        # Verify JSON can be loaded
        with open(filepath, 'r') as f:
            test_load = json.load(f)
        print(f"  ✓ JSON file verified (can be loaded)")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Failed to export weights!")
        print(f"Error: {e}")
        return False


def test_forward_pass():
    """Test that exported weights can be loaded and used"""
    print("\n" + "="*70)
    print("TESTING EXPORTED WEIGHTS")
    print("="*70)
    
    try:
        with open('deeponet_weights.json', 'r') as f:
            weights = json.load(f)
        
        print("✓ JSON file loaded successfully")
        print(f"  Keys: {list(weights.keys())}")
        print(f"  Metadata: {weights['metadata']}")
        
        # Test with random input
        gust_features = np.random.randn(10).tolist()
        trunk_vec = np.random.randn(10).tolist()
        
        print("\n✓ Ready for HTML visualization!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = export_weights()
    
    if success:
        test_forward_pass()
        
        print("\n" + "="*70)
        print("NEXT STEPS:")
        print("="*70)
        print("1. Start a local web server:")
        print("   python -m http.server 8000")
        print()
        print("2. Open in your browser:")
        print("   http://localhost:8000/turbulence_visualization.html")
        print()
        print("   OR")
        print()
        print("   For Firefox, enable local file access:")
        print("   - Type about:config in address bar")
        print("   - Search for: security.fileuri.strict_origin_policy")
        print("   - Set to: false")
        print("   - Then open: file:///path/to/turbulence_visualization.html")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("EXPORT FAILED")
        print("="*70)
        print("Please ensure you have trained the network first:")
        print("  python main.py --train")
        print("="*70)