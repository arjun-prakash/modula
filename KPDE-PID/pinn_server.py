#!/usr/bin/env python3
"""
pinn_server.py - Flask server for real-time PINN predictions

This server provides an HTTP API endpoint that the HTML visualization
can call to get real PINN predictions from the trained network.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from deeponet import DeepONet
from controllers import encode_gust_features, encode_trunk_vec
from physics_models import GustField
import sys

app = Flask(__name__)
CORS(app)  # Enable CORS for browser requests

# Global network instance
network = None
network_loaded = False


def load_network():
    """Load the trained DeepONet"""
    global network, network_loaded
    
    try:
        network = DeepONet()
        network.load_weights("deeponet_weights.npz")
        network_loaded = True
        print("✓ DeepONet weights loaded successfully!")
        return True
    except Exception as e:
        print(f"✗ Failed to load network: {e}")
        print("Make sure deeponet_weights.npz exists in this directory")
        return False


@app.route('/')
def index():
    """Status page"""
    status = "✓ Ready" if network_loaded else "✗ Not loaded"
    return f"""
    <html>
    <head><title>PINN Server</title></head>
    <body style="font-family: monospace; padding: 20px;">
        <h1>🚁 PINN Prediction Server</h1>
        <p><strong>Status:</strong> {status}</p>
        <p><strong>Endpoint:</strong> POST /predict</p>
        <hr>
        <h3>Usage:</h3>
        <pre>
POST /predict
Content-Type: application/json

{{
  "pos": [x, y, z],
  "vel": [vx, vy, vz],
  "bursts": [...],
  "time": t,
  "base_wind": [wx, wy, wz]
}}
        </pre>
        <hr>
        <p><em>Server running on http://localhost:5000</em></p>
    </body>
    </html>
    """


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict disturbance force using trained PINN
    
    Request JSON format:
    {
        "pos": [x, y, z],           # Drone position
        "vel": [vx, vy, vz],        # Drone velocity
        "bursts": [                 # List of active gust bursts
            {
                "position": [x, y, z],
                "intensity": float,
                "lifetime": float
            },
            ...
        ],
        "time": float,              # Current simulation time
        "base_wind": [wx, wy, wz]   # Base wind vector (optional)
    }
    
    Response JSON format:
    {
        "prediction": [fx, fy, fz],  # Predicted disturbance force
        "status": "success"
    }
    """
    if not network_loaded:
        return jsonify({
            'error': 'Network not loaded',
            'status': 'error'
        }), 500
    
    try:
        data = request.json
        
        # Extract data
        pos = np.array(data['pos'], dtype=np.float32)
        vel = np.array(data['vel'], dtype=np.float32)
        bursts_data = data['bursts']
        time = float(data['time'])
        base_wind = np.array(data.get('base_wind', [0, 0, 0]), dtype=np.float32)
        
        # Create temporary gust field for encoding
        temp_field = GustField(max_bursts=len(bursts_data))
        temp_field.bursts = []
        
        for burst_data in bursts_data[:2]:  # Only use first 2 bursts
            temp_field.bursts.append({
                'position': np.array(burst_data['position'], dtype=np.float32),
                'intensity': float(burst_data['intensity']),
                'lifetime': float(burst_data.get('lifetime', 1.0)),
                'pulse_freq': float(burst_data.get('pulse_freq', 5.0))
            })
        
        # Add base wind to field
        temp_field.base_wind = base_wind
        temp_field.storm_mode = np.linalg.norm(base_wind) > 1.0
        
        # Encode features
        state = np.concatenate([pos, vel]).astype(np.float32)
        gust_features = encode_gust_features(pos, temp_field, time, max_bursts=2)
        trunk_vec = encode_trunk_vec(state, temp_field, time)
        
        # Get prediction
        prediction = network.forward(
            gust_features.reshape(1, -1),
            trunk_vec.reshape(1, -1)
        )[0]
        
        return jsonify({
            'prediction': prediction.tolist(),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if network_loaded else 'unhealthy',
        'network_loaded': network_loaded
    })


if __name__ == '__main__':
    print("="*70)
    print("PINN PREDICTION SERVER")
    print("="*70)
    print()
    
    # Load network
    if not load_network():
        print("\n✗ Failed to start server - network not loaded")
        print("Please train the network first:")
        print("  python main.py --train")
        sys.exit(1)
    
    print()
    print("="*70)
    print("SERVER READY")
    print("="*70)
    print("URL: http://localhost:5000")
    print("API Endpoint: POST http://localhost:5000/predict")
    print()
    print("Press Ctrl+C to stop")
    print("="*70)
    print()
    
    # Start server
    app.run(
        host='0.0.0.0',  # Accept connections from any IP
        port=5000,
        debug=False,     # Set to True for development
        threaded=True    # Handle multiple requests
    )