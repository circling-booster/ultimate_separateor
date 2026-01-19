import os
import sys
import time
import yaml
import librosa
from dotenv import load_dotenv
from src.pipeline import ProcessingPipeline
from src.utils import setup_logger, get_files, fmt_time, save_process_info

def load_config():
    load_dotenv() # Load .env
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Merge env into a dict for easy access
    env_vars = {
        'INPUT_DIR': os.getenv('INPUT_DIR', './files'),
        'OUTPUT_DIR': os.getenv('OUTPUT_DIR', './studio_output'),
        'MODEL_DIR': os.getenv('MODEL_DIR', './models'),
        'TARGET_FILENAME': os.getenv('TARGET_FILENAME', ''),
        'COOLDOWN': int(os.getenv('COOLDOWN_SECONDS', 1)),
        'SAVE_ALL_ARTIFACTS': os.getenv('SAVE_ALL_ARTIFACTS', 'True')
    }
    return config, env_vars

def main():
    setup_logger()
    config, env = load_config()
    
    print("="*60)
    print(f"🛠️  Studio Separator Refactored")
    print(f"📂 Input: {env['INPUT_DIR']}")
    print(f"🎯 Target: {env['TARGET_FILENAME'] if env['TARGET_FILENAME'] else 'All Files'}")
    print("="*60 + "\n")

    files = get_files(
        env['INPUT_DIR'], 
        {'.mp3', '.flac', '.wav', '.m4a', '.aiff'}, 
        env['TARGET_FILENAME']
    )

    if not files:
        print("❌ No audio files found. Please check 'files' folder or .env settings.")
        return

    pipeline = ProcessingPipeline(config, env)

    for idx, file_path in enumerate(files):
        print(f"🎵 Processing [{idx+1}/{len(files)}]: {os.path.basename(file_path)}")
        start_time = time.time()
        
        try:
            duration = librosa.get_duration(path=file_path)
            output_dir = pipeline.run(file_path, env['OUTPUT_DIR'])
            
            elapsed = time.time() - start_time
            save_process_info(output_dir, os.path.basename(file_path), duration, elapsed, config)
            
            print(f"✅ Completed in {fmt_time(elapsed)}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

        # Cooldown
        if idx < len(files) - 1:
            print(f"💤 Cooling down {env['COOLDOWN']}s...")
            time.sleep(env['COOLDOWN'])

if __name__ == "__main__":
    main()