import os
import shutil
import time
import soundfile as sf
import librosa
from pathlib import Path
from .inference import InferenceEngine
from .audio_ops import normalize_peak, align_and_mix, smart_gate, mix_weighted

class ProcessingPipeline:
    def __init__(self, config, env):
        self.cfg = config
        self.env = env
        self.engine = InferenceEngine(
            model_dir=env['MODEL_DIR'],
            temp_format=config['work_settings']['temp_format'],
            params=config['inference_params']
        )

    def run(self, input_path, output_root_dir):
        file_stem = Path(input_path).stem
        song_out_dir = os.path.join(output_root_dir, file_stem)
        raw_dir = os.path.join(song_out_dir, "Raw_Artifacts")
        
        # 디렉토리 생성
        Path(song_out_dir).mkdir(parents=True, exist_ok=True)
        Path(raw_dir).mkdir(parents=True, exist_ok=True)

        print(f"    🔄 Checking for existing checkpoints...")

        # ---------------------------------------------------------
        # [Step 0] Pre-processing (Normalization)
        # ---------------------------------------------------------
        norm_dir = os.path.join(raw_dir, "00_Normalization")
        norm_file = os.path.join(norm_dir, "00_Normalized.wav")
        current_input = input_path

        if self.cfg['work_settings']['enable_normalization']:
            if os.path.exists(norm_file):
                print(f"    [0/4] ✅ Found normalized file. Skipping...")
                current_input = norm_file
            else:
                print(f"    [0/4] Pre-processing (Normalization)...")
                Path(norm_dir).mkdir(exist_ok=True)
                current_input = normalize_peak(input_path, norm_file, self.cfg['work_settings']['target_peak_db'])
        
        y_orig, sr = librosa.load(current_input, sr=None, mono=False)

        # ---------------------------------------------------------
        # [Step 1] Vocal Separation (Ensemble)
        # ---------------------------------------------------------
        ens_path = os.path.join(raw_dir, '01_Ensemble_Vocals.wav')
        
        if os.path.exists(ens_path):
             print(f"    [1/4] ✅ Found ensemble vocals. Skipping...")
        else:
            print(f"    [1/4] Vocal Separation (Ensemble)...")
            models = [
                ("RoFormer", self.cfg['models']['vocal_1'], "01_A_RoF"),
                ("MDX", self.cfg['models']['vocal_2'], "01_B_MDX")
            ]
            
            stem_paths = []
            for name, model_file, sub_dir in models:
                work_dir = os.path.join(raw_dir, sub_dir)
                
                existing_voc = None
                if os.path.exists(work_dir):
                    existing_files = os.listdir(work_dir)
                    found = [f for f in existing_files if "Vocals" in f and f.endswith(".wav")]
                    if found:
                        existing_voc = os.path.join(work_dir, found[0])

                if existing_voc:
                    print(f"        🔹 Found {name} output. Using cached file.")
                    stem_paths.append(existing_voc)
                else:
                    print(f"        🔹 Running {name}...")
                    out_files = self.engine.separate(current_input, work_dir, model_file)
                    voc_path = self.engine.find_file_by_keyword(out_files, "Vocals", work_dir)
                    stem_paths.append(voc_path)

            print(f"        🔹 Mixing...")
            voc_ens, _ = align_and_mix(stem_paths, self.cfg['weights']['ensemble_2_model']['Balanced'], sr)
            min_len = min(y_orig.shape[-1], voc_ens.shape[-1])
            voc_ens = voc_ens[..., :min_len]
            sf.write(ens_path, voc_ens.T, sr)

        # ---------------------------------------------------------
        # [Step 2] De-Reverb (Optimized for 512 Dim)
        # ---------------------------------------------------------
        dry_final_path = os.path.join(raw_dir, '02_Dry_Vocals.wav')
        
        if os.path.exists(dry_final_path):
            print(f"    [2/4] ✅ Found dry vocals. Skipping...")
        else:
            print(f"    [2/4] De-reverberation...")
            dr_dir = os.path.join(raw_dir, "02_DeReverb")
            
            out_dr_files = []
            if os.path.exists(dr_dir):
                 out_dr_files = [f for f in os.listdir(dr_dir) if "No_Reverb" in f]
            
            if out_dr_files:
                 dry_path = os.path.join(dr_dir, out_dr_files[0])
                 shutil.copy(dry_path, dry_final_path)
            else:
                 # [CRITICAL FIX] Reverb 모델 확인 결과 512가 맞음. 
                 # 512를 강제 주입하여 불일치 방지.
                 try:
                     reverb_overrides = {
                         "mdx_params": {
                             "segment_size": 512,  # Fixed: 256 -> 512
                             "overlap": 0.25, 
                             "batch_size": 1
                         }
                     }
                     out_dr = self.engine.separate(
                         ens_path, 
                         dr_dir, 
                         self.cfg['models']['dereverb'],
                         **reverb_overrides
                     )
                     dry_path = self.engine.find_file_by_keyword(out_dr, "No_Reverb", dr_dir)
                     shutil.copy(dry_path, dry_final_path)
                 except Exception as e:
                     print(f"        ⚠️  Reverb Failed: {e}")
                     print(f"        ⚠️  Skipping De-Reverb step (Using Ensemble Vocals as Dry).")
                     shutil.copy(ens_path, dry_final_path)

        # ---------------------------------------------------------
        # [Step 3] Smart Gate
        # ---------------------------------------------------------
        clean_path = os.path.join(raw_dir, '03_Clean_Vocals.wav')
        final_dest = os.path.join(song_out_dir, '03_vocals_final.wav')
        
        if os.path.exists(clean_path) and os.path.exists(final_dest):
             print(f"    [3/4] ✅ Found clean vocals. Skipping...")
        else:
            print(f"    [3/4] Smart Gate...")
            smart_gate(dry_final_path, clean_path, sr)
            shutil.copy(clean_path, final_dest)

        # ---------------------------------------------------------
        # [Step 4] Karaoke Split (Fixed UVR-MDX-Karaoke)
        # ---------------------------------------------------------
        main_out = os.path.join(song_out_dir, '04_main_vocal.wav')
        back_out = os.path.join(song_out_dir, '04_backing_vocal.wav')
        
        if self.cfg['work_settings']['enable_karaoke']:
            if os.path.exists(main_out) and os.path.exists(back_out):
                print(f"    [4/4] ✅ Found karaoke split. Skipping...")
            else:
                print(f"    [4/4] Main/Backing Split...")
                
                # 1. RoFormer (Robust)
                dir_rof = os.path.join(raw_dir, "04_A_RoF")
                main_rof, back_rof = self._get_or_run_karaoke(
                    clean_path, dir_rof, self.cfg['models']['karaoke_rof'],
                    is_mdx=False
                )

                # 2. MDX (Needs 512 override)
                dir_mdx = os.path.join(raw_dir, "04_B_MDX")
                try:
                    main_mdx, back_mdx = self._get_or_run_karaoke(
                        clean_path, dir_mdx, self.cfg['models']['karaoke_mdx'],
                        is_mdx=True
                    )
                    
                    # Both successful -> Mix
                    print(f"        🔹 Mixing Karaoke Stems...")
                    w_rof = self.cfg['weights']['karaoke']['RoFormer']
                    w_mdx = self.cfg['weights']['karaoke']['MDX']
                    
                    main_final, _ = mix_weighted(main_rof, main_mdx, w_rof, w_mdx, sr)
                    back_final, _ = mix_weighted(back_rof, back_mdx, w_rof, w_mdx, sr)

                except Exception as e:
                    print(f"        ⚠️  MDX Karaoke Failed: {e}")
                    print(f"        ⚠️  Falling back to RoFormer output only.")
                    main_final, _ = librosa.load(main_rof, sr=sr, mono=False)
                    back_final, _ = librosa.load(back_rof, sr=sr, mono=False)
                    if main_final.ndim == 1: main_final = np.stack([main_final, main_final])
                    if back_final.ndim == 1: back_final = np.stack([back_final, back_final])

                sf.write(main_out, main_final.T, sr)
                sf.write(back_out, back_final.T, sr)
        else:
            print("    [4/4] Skipped (Disabled).")

        if str(self.env.get('SAVE_ALL_ARTIFACTS')).lower() != 'true':
            pass
            
        return song_out_dir

    def _get_or_run_karaoke(self, input_path, work_dir, model_file, is_mdx=False):
        existing_files = []
        if os.path.exists(work_dir):
            existing_files = [os.path.join(work_dir, f) for f in os.listdir(work_dir) if f.endswith('.wav')]
        
        if len(existing_files) >= 2:
            print(f"        🔹 Using cached output from {os.path.basename(work_dir)}...")
            return self.engine.identify_stems([os.path.basename(f) for f in existing_files], work_dir, input_path)
        else:
            print(f"        🔹 Running {os.path.basename(work_dir)}...")
            
            # [FIX] MDX Karaoke 모델도 512가 필요함
            overrides = {}
            if is_mdx:
                overrides = {
                    "mdx_params": {
                        "segment_size": 512, 
                        "overlap": 0.25, 
                        "batch_size": 1
                    }
                }
                
            out_files = self.engine.separate(input_path, work_dir, model_file, **overrides)
            return self.engine.identify_stems(out_files, work_dir, input_path)