import os
import sys
import pickle
import json

# Setup module alias so pickle can load classes from 'FakeApp'
import GenuineApp
import GenuineApp.ml_engine
sys.modules['FakeApp'] = GenuineApp
sys.modules['FakeApp.ml_engine'] = GenuineApp.ml_engine

def convert_np(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    import numpy as np
    if isinstance(obj, dict):
        return {convert_np(k): convert_np(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_np(x) for x in obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_np(obj.tolist())
    elif isinstance(obj, np.str_):
        return str(obj)
    else:
        return obj

def main():
    model_dir = os.path.join(os.path.dirname(__file__), 'model')
    pkl_path = os.path.join(model_dir, 'detector.pkl')
    json_path = os.path.join(model_dir, 'metrics.json')

    if not os.path.exists(pkl_path):
        print(f"[-] detector.pkl not found at {pkl_path}. Cannot extract metrics.")
        sys.exit(1)

    print(f"[*] Loading {pkl_path} to extract metrics...")
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        metrics = data.get('metrics', {})
        if not metrics:
            print("[-] No metrics found in detector.pkl.")
            sys.exit(1)

        print("[*] Converting numpy types to native Python types...")
        clean_metrics = convert_np(metrics)

        print(f"[*] Saving metrics to {json_path}...")
        with open(json_path, 'w') as f:
            json.dump(clean_metrics, f, indent=4)
        
        print("[+] Metrics successfully extracted and saved to metrics.json!")
    except Exception as e:
        print(f"[-] Error extracting metrics: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
