#!/usr/bin/env python3
"""
Unified GPU toolkit for training LLMs
Combines hardware analysis, configuration optimization, and memory management
"""

import os
import sys
import argparse
from pathlib import Path

# Import from other scripts
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))

try:
    from config_optimizer import get_system_info, benchmark_gpu, generate_recommended_config, print_recommendations
    from optimize_gpu_memory import print_gpu_memory_stats, optimize_memory_settings, run_memory_test
    has_imports = True
except ImportError as e:
    print(f"Warning: Could not import all modules: {e}")
    has_imports = False

def get_config_file_path(config_name):
    """Get the full path for a configuration file"""
    if os.path.isfile(config_name):
        return config_name
    
    # Try to find in config directory
    config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
    candidate = os.path.join(config_dir, config_name)
    if os.path.isfile(candidate):
        return candidate
    
    # Try adding yaml extension
    if not config_name.endswith('.yaml'):
        candidate = os.path.join(config_dir, f"{config_name}.yaml")
        if os.path.isfile(candidate):
            return candidate
    
    return config_name  # Return original if not found

def list_configs():
    """List available configuration files"""
    config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
    if not os.path.isdir(config_dir):
        return []
    
    return [f for f in os.listdir(config_dir) if f.endswith('.yaml') and f.startswith('text_generation')]

def print_available_configs():
    """Print available configuration files with descriptions"""
    configs = list_configs()
    if not configs:
        print("No configuration files found")
        return
    
    print("Available Configurations:")
    print("-------------------------")
    
    descriptions = {
        "text_generation.yaml": "Default configuration (16GB+ VRAM)",
        "text_generation_optimized.yaml": "Optimized configuration (10GB+ VRAM)",
        "text_generation_optimized_memory.yaml": "Memory-optimized configuration (8-12GB VRAM)",
        "text_generation_low_memory.yaml": "Low-memory configuration (6-8GB VRAM)",
        "text_generation_sanity_test.yaml": "Sanity test configuration (minimal resources)"
    }
    
    for config in sorted(configs):
        desc = descriptions.get(config, "")
        print(f"- {config:<35} {desc}")

def show_command_help(name):
    """Show detailed help for a specific command"""
    commands = {
        "analyze": """
Analyze your hardware and recommend configuration settings:

python scripts/gpu_tools.py analyze
python scripts/gpu_tools.py analyze --config text_generation.yaml
python scripts/gpu_tools.py analyze --update --config text_generation.yaml
python scripts/gpu_tools.py analyze --output my_config.yaml
        """,
        
        "memory": """
Check GPU memory usage and run memory tests:

python scripts/gpu_tools.py memory
python scripts/gpu_tools.py memory --test
python scripts/gpu_tools.py memory --stats
        """,
        
        "configs": """
List available configuration files:

python scripts/gpu_tools.py configs
python scripts/gpu_tools.py configs --describe
        """,
        
        "recommend": """
Get a recommendation for which configuration to use:

python scripts/gpu_tools.py recommend
python scripts/gpu_tools.py recommend --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
        """
    }
    
    if name in commands:
        print(f"Help for '{name}':")
        print(commands[name])
    else:
        print(f"No detailed help available for '{name}'")

def recommend_config():
    """Recommend which configuration file to use based on hardware"""
    if not has_imports:
        print("Cannot make recommendations: required modules not available")
        return
    
    try:
        sys_info = get_system_info()
        
        if not sys_info["gpu"]["available"]:
            print("\n⚠️ No GPU detected. Use CPU-only configuration or cloud GPU.")
            return
        
        # Get GPU memory 
        gpu_memory = 0
        if sys_info["gpu"]["devices"] and sys_info["gpu"]["devices"][0].get("memory_gb"):
            gpu_memory = sys_info["gpu"]["devices"][0]["memory_gb"]
        
        print(f"\nDetected GPU: {sys_info['gpu']['devices'][0]['name']} with {gpu_memory:.1f} GB VRAM")
        
        if gpu_memory >= 16:
            print("Recommendation: Use default configuration")
            print("Command: make recipe-train DATA_DIR=~/recipe_manual_data")
        elif gpu_memory >= 12:
            print("Recommendation: Use memory-optimized configuration")
            print("Command: make recipe-train-high-memory DATA_DIR=~/recipe_manual_data") 
        elif gpu_memory >= 8:
            print("Recommendation: Use optimized configuration with reduced dataset")
            print("Command: make recipe-train-optimized DATA_DIR=~/recipe_manual_data")
        elif gpu_memory >= 6:
            print("Recommendation: Use low-memory configuration")
            print("Command: make recipe-train-low-memory DATA_DIR=~/recipe_manual_data")
        else:
            print("⚠️ Your GPU has very limited memory.")
            print("Recommendation: Use sanity test configuration or CPU training")
            print("Command: make recipe-train-test DATA_DIR=~/recipe_manual_data")
        
    except Exception as e:
        print(f"Error generating recommendation: {e}")

def main():
    parser = argparse.ArgumentParser(description="GPU toolkit for LLM training")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze hardware and recommend configurations')
    analyze_parser.add_argument('--config', type=str, default='config/text_generation.yaml',
                        help='Path to configuration file')
    analyze_parser.add_argument('--update', action='store_true',
                        help='Update the config file with recommendations')
    analyze_parser.add_argument('--output', type=str, default=None,
                        help='Path to save updated config (if not overwriting)')
    
    # Memory command
    memory_parser = subparsers.add_parser('memory', help='GPU memory tools')
    memory_parser.add_argument('--stats', action='store_true', help='Display GPU memory statistics')
    memory_parser.add_argument('--test', action='store_true', help='Run memory allocation test')
    memory_parser.add_argument('--env', action='store_true', help='Show recommended environment variables')
    memory_parser.add_argument('--batch', action='store_true', help='Suggest batch size based on GPU memory')
    
    # Configs command
    configs_parser = subparsers.add_parser('configs', help='List available configurations')
    configs_parser.add_argument('--describe', action='store_true', help='Show config descriptions')
    
    # Recommend command
    recommend_parser = subparsers.add_parser('recommend', help='Get config recommendation')
    recommend_parser.add_argument('--model', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0', 
                                 help='Model to analyze')
    
    # Help command
    help_parser = subparsers.add_parser('help', help='Show detailed help for a command')
    help_parser.add_argument('command_name', nargs='?', default='', help='Command name to get help for')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
        
    if args.command == 'analyze':
        if not has_imports:
            print("Cannot analyze: config_optimizer.py functionality not available")
            return
            
        config_path = get_config_file_path(args.config)
        
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
        import yaml
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
    
    elif args.command == 'memory':
        default_all = not (args.stats or args.test or args.env or args.batch)
        
        if args.stats or default_all:
            print_gpu_memory_stats()
        
        if args.env or default_all:
            print("\nRecommended environment variable:")
            print(optimize_memory_settings())
            print("\nUsage example:")
            print(f"{optimize_memory_settings()} python -m src.model.recipe_train --config config/text_generation_optimized.yaml")
        
        if args.batch or default_all:
            print("\nRecommended batch size for your GPU:", suggest_batch_size())
        
        if args.test:
            run_memory_test()
    
    elif args.command == 'configs':
        print_available_configs()
    
    elif args.command == 'recommend':
        recommend_config()
    
    elif args.command == 'help':
        if args.command_name:
            show_command_help(args.command_name)
        else:
            parser.print_help()

if __name__ == '__main__':
    main()
