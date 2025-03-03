#!/usr/bin/env python3
"""
Hardware configuration optimizer for LLM fine-tuning
Analyzes system resources and recommends optimal training settings
"""

import os
import sys
import psutil
import yaml
import argparse
from pathlib import Path
import math

# Try importing GPU-related libraries
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import py3nvml.py3nvml as nvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False

def get_system_info():
    """Gather system hardware information"""
    info = {
        "cpu": {
            "cores": psutil.cpu_count(logical=False),
            "threads": psutil.cpu_count(logical=True),
            "memory_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2)
        },
        "gpu": {
            "available": False,
            "count": 0,
            "devices": []
        }
    }
    
    # Check for GPU availability
    if HAS_TORCH and torch.cuda.is_available():
        info["gpu"]["available"] = True
        info["gpu"]["count"] = torch.cuda.device_count()
        
        if HAS_NVML:
            try:
                nvml.nvmlInit()
                for i in range(info["gpu"]["count"]):
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)
                    device_info = {
                        "name": nvml.nvmlDeviceGetName(handle),
                        "memory_gb": round(nvml.nvmlDeviceGetMemoryInfo(handle).total / (1024 ** 3), 2)
                    }
                    info["gpu"]["devices"].append(device_info)
                nvml.nvmlShutdown()
            except:
                # Fall back to torch info if NVML fails
                for i in range(info["gpu"]["count"]):
                    device_info = {
                        "name": torch.cuda.get_device_name(i),
                        "memory_gb": None  # Can't get memory info with torch alone
                    }
                    info["gpu"]["devices"].append(device_info)
        else:
            # Use torch info if NVML is not available
            for i in range(info["gpu"]["count"]):
                device_info = {
                    "name": torch.cuda.get_device_name(i),
                    "memory_gb": None
                }
                info["gpu"]["devices"].append(device_info)
                
    return info

def estimate_model_size(model_name):
    """Estimate model size based on name or provide defaults"""
    # Very rough estimates for common model sizes
    if "llama-7b" in model_name.lower():
        return 7
    elif "llama-13b" in model_name.lower():
        return 13
    elif "llama-70b" in model_name.lower():
        return 70
    elif "tinyllama" in model_name.lower():
        return 1.1
    elif "mistral" in model_name.lower() and "7b" in model_name.lower():
        return 7
    else:
        # Try to extract size from name
        import re
        match = re.search(r'(\d+)[bB]', model_name)
        if match:
            return float(match.group(1))
        # Default estimate
        return 1.1  # Assume TinyLlama size if unknown

def benchmark_gpu(model_name):
    """Simple GPU benchmark for model loading and inference"""
    if not HAS_TORCH or not torch.cuda.is_available():
        return None
    
    results = {
        "can_load_model": False,
        "recommended_batch_size": 1,
        "recommended_precision": "fp16"
    }
    
    # Estimate model size and memory requirements
    model_size_gb = estimate_model_size(model_name)
    
    # Get GPU memory
    gpu_memory = 0
    if HAS_NVML:
        try:
            nvml.nvmlInit()
            handle = nvml.nvmlDeviceGetHandleByIndex(0)  # Use primary GPU
            gpu_memory = nvml.nvmlDeviceGetMemoryInfo(handle).total / (1024 ** 3)
            nvml.nvmlShutdown()
        except:
            pass
    
    if gpu_memory == 0:
        # If we couldn't get memory info, make a conservative estimate
        gpu_memory = 8  # Assume 8GB as a safe default
    
    # Determine if model can fit in memory with LoRA
    if model_size_gb * 0.4 < gpu_memory:  # LoRA reduces memory by ~60%
        results["can_load_model"] = True
        
        # Calculate recommended batch size based on available memory
        # Leave 20% of memory for overhead
        available_memory = gpu_memory * 0.8
        memory_per_sample = (model_size_gb * 0.4) * 1.5  # LoRA + 50% for activations
        
        max_batch_size = max(1, int(available_memory / memory_per_sample))
        
        # Be conservative with batch size recommendations
        if max_batch_size >= 8:
            results["recommended_batch_size"] = 8
        elif max_batch_size >= 4:
            results["recommended_batch_size"] = 4
        elif max_batch_size >= 2:
            results["recommended_batch_size"] = 2
        else:
            results["recommended_batch_size"] = 1
            
        # Set gradient accumulation to achieve effective batch size of ~32
        results["gradient_accumulation"] = max(1, 32 // results["recommended_batch_size"])
        
        # Determine best precision
        if "A100" in str(torch.cuda.get_device_name(0)) or "H100" in str(torch.cuda.get_device_name(0)):
            results["recommended_precision"] = "bf16"
        elif "V100" in str(torch.cuda.get_device_name(0)) or "T4" in str(torch.cuda.get_device_name(0)):
            results["recommended_precision"] = "fp16"
        else:
            # For older GPUs, recommend fp16 or fp32 based on memory
            if gpu_memory > 16:
                results["recommended_precision"] = "fp16"
            else:
                results["recommended_precision"] = "fp32"
                
    return results

def generate_recommended_config(sys_info, config_path):
    """Generate recommended config based on hardware and existing config"""
    # Load existing config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_name = config.get('model', {}).get('name', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    
    # Generate recommendations
    recommendations = {
        "use_gpu": sys_info["gpu"]["available"],
        "cpu_threads": sys_info["cpu"]["threads"],
        "max_train_samples": None
    }
    
    # If GPU is available, get detailed recommendations
    if recommendations["use_gpu"]:
        gpu_benchmark = benchmark_gpu(model_name)
        if gpu_benchmark:
            recommendations.update(gpu_benchmark)
    
    # Memory-based dataset size recommendations
    system_memory_gb = sys_info["cpu"]["memory_gb"]
    if system_memory_gb < 16:
        recommendations["max_train_samples"] = 10000
    elif system_memory_gb < 32:
        recommendations["max_train_samples"] = 50000
    
    # Update config with recommendations
    updated_config = config.copy()
    
    # Update training settings
    if recommendations["use_gpu"]:
        if gpu_benchmark and gpu_benchmark["can_load_model"]:
            updated_config["training"]["batch_size"] = gpu_benchmark["recommended_batch_size"]
            updated_config["training"]["gradient_accumulation_steps"] = gpu_benchmark["gradient_accumulation"]
            
            if gpu_benchmark["recommended_precision"] == "fp16":
                updated_config["training"]["fp16"] = True
                if "bf16" in updated_config["training"]:
                    updated_config["training"]["bf16"] = False
            elif gpu_benchmark["recommended_precision"] == "bf16":
                updated_config["training"]["fp16"] = False
                updated_config["training"]["bf16"] = True
            elif gpu_benchmark["recommended_precision"] == "fp32":
                updated_config["training"]["fp16"] = False
                if "bf16" in updated_config["training"]:
                    updated_config["training"]["bf16"] = False
    
    # Update dataset size if needed
    if recommendations["max_train_samples"]:
        if "max_train_samples" not in updated_config["data"]:
            updated_config["data"]["max_train_samples"] = recommendations["max_train_samples"]
    
    return updated_config, recommendations

def print_recommendations(sys_info, recommendations, original_config, updated_config):
    """Print recommendations in a readable format"""
    print("\n" + "="*50)
    print("📊 SYSTEM INFORMATION")
    print("="*50)
    print(f"CPU: {sys_info['cpu']['cores']} cores / {sys_info['cpu']['threads']} threads")
    print(f"RAM: {sys_info['cpu']['memory_gb']} GB")
    
    if sys_info["gpu"]["available"]:
        print("\nGPU Information:")
        for i, device in enumerate(sys_info["gpu"]["devices"]):
            print(f"  GPU {i}: {device['name']}" + 
                  (f" ({device['memory_gb']} GB)" if device['memory_gb'] else ""))
    else:
        print("\nNo GPU detected. Training will be slow on CPU only.")
    
    print("\n" + "="*50)
    print("💡 RECOMMENDED CONFIGURATION")
    print("="*50)
    
    model_name = original_config.get('model', {}).get('name', '')
    print(f"Model: {model_name}")
    
    if recommendations["use_gpu"]:
        if recommendations.get("can_load_model", False):
            print("\n✅ Your GPU can handle this model with LoRA fine-tuning")
            print(f"Recommended batch size: {recommendations.get('recommended_batch_size', 1)}")
            print(f"Gradient accumulation: {recommendations.get('gradient_accumulation', 32)}")
            print(f"Effective batch size: {recommendations.get('recommended_batch_size', 1) * recommendations.get('gradient_accumulation', 32)}")
            print(f"Precision: {recommendations.get('recommended_precision', 'fp16')}")
        else:
            print("\n⚠️ This model may be too large for your GPU even with LoRA")
            print("Consider using a smaller model or quantization")
    else:
        print("\n⚠️ No GPU available, CPU training will be very slow")
    
    if recommendations.get("max_train_samples"):
        print(f"\nRecommended dataset limit: {recommendations['max_train_samples']} samples")
        print("(This helps prevent memory issues when loading the dataset)")
    
    print("\n" + "="*50)
    print("🛠️ CONFIGURATION CHANGES")
    print("="*50)
    
    # Show key differences
    if original_config["training"].get("batch_size") != updated_config["training"].get("batch_size"):
        print(f"batch_size: {original_config['training'].get('batch_size')} → {updated_config['training'].get('batch_size')}")
    
    if original_config["training"].get("gradient_accumulation_steps") != updated_config["training"].get("gradient_accumulation_steps"):
        print(f"gradient_accumulation_steps: {original_config['training'].get('gradient_accumulation_steps')} → {updated_config['training'].get('gradient_accumulation_steps')}")
    
    fp16_changed = original_config["training"].get("fp16") != updated_config["training"].get("fp16")
    bf16_changed = original_config["training"].get("bf16", None) != updated_config["training"].get("bf16", None)
    
    if fp16_changed or bf16_changed:
        old_precision = "fp16" if original_config["training"].get("fp16") else ("bf16" if original_config["training"].get("bf16", False) else "fp32")
        new_precision = "fp16" if updated_config["training"].get("fp16") else ("bf16" if updated_config["training"].get("bf16", False) else "fp32")
        print(f"precision: {old_precision} → {new_precision}")
    
    if ("max_train_samples" not in original_config.get("data", {}) and 
        "max_train_samples" in updated_config.get("data", {})):
        print(f"max_train_samples: none → {updated_config['data']['max_train_samples']}")

def main():
    parser = argparse.ArgumentParser(description='Analyze hardware and optimize LLM training configuration')
    parser.add_argument('--config', type=str, default='config/text_generation.yaml',
                        help='Path to configuration file')
    parser.add_argument('--update', action='store_true',
                        help='Update the config file with recommendations')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save updated config (if not overwriting)')
    args = parser.parse_args()
    
    # Ensure config file exists
    config_path = args.config
    if not os.path.isfile(config_path):
        print(f"Error: Config file not found: {config_path}")
        return 1
    
    # Get system information
    print("Analyzing system hardware...")
    sys_info = get_system_info()
    
    # Generate recommended config
    print("Generating recommendations...")
    updated_config, recommendations = generate_recommended_config(sys_info, config_path)
    
    # Load original config for comparison
    with open(config_path, 'r') as f:
        original_config = yaml.safe_load(f)
    
    # Print recommendations
    print_recommendations(sys_info, recommendations, original_config, updated_config)
    
    # Save updated config if requested
    if args.update:
        output_path = args.output if args.output else config_path
        print(f"\nSaving updated config to {output_path}")
        with open(output_path, 'w') as f:
            yaml.dump(updated_config, f, default_flow_style=False)
        print("✅ Configuration updated successfully")
    elif args.output:
        print(f"\nSaving recommended config to {args.output}")
        with open(args.output, 'w') as f:
            yaml.dump(updated_config, f, default_flow_style=False)
        print("✅ Recommended configuration saved successfully")
    else:
        print("\nTo save these recommendations, run with --update flag or specify --output")

if __name__ == "__main__":
    sys.exit(main())
