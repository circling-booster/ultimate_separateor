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
        
        Path(song_out_dir).mkdir(parents=True, exist_ok=True)
        Path(raw_dir).mkdir(parents=True, exist_ok=True)

        print(f"    [0/4] Pre-processing...")
        current_input = input_path
        
        # 0. Normalization
        if self.cfg['work_settings']['enable_normalization']:
            norm_dir = os.path.join(raw_dir, "00_Normalization")
            Path(norm_dir).mkdir(exist_ok=True)
            norm_file = os.path.join(norm_dir, "00_Normalized.wav")
            current_input = normalize_peak(input_path, norm_file, self.cfg['work_settings']['target_peak_db'])

        y_orig, sr = librosa.load(current_input, sr=None, mono=False)

        # 1. Vocal Separation (Ensemble)
        print(f"    [1/4] Vocal Separation...")
        models = [
            ("RoFormer", self.cfg['models']['vocal_1'], "01_A_RoF"),
            ("MDX", self.cfg['models']['vocal_2'], "01_B_MDX")
        ]
        
        stem_paths = []
        for name, model_file, sub_dir in models:
            work_dir = os.path.join(raw_dir, sub_dir)
            out_files = self.engine.separate(current_input, work_dir, model_file)
            voc_path = self.engine.find_file_by_keyword(out_files, "Vocals", work_dir)
            stem_paths.append(voc_path)

        # Mix Vocals
        voc_ens, _ = align_and_mix(stem_paths, self.cfg['weights']['ensemble_2_model']['Balanced'], sr)
        
        # Length Fix
        min_len = min(y_orig.shape[-1], voc_ens.shape[-1])
        voc_ens = voc_ens[..., :min_len]
        
        ens_path = os.path.join(raw_dir, '01_Ensemble_Vocals.wav')
        sf.write(ens_path, voc_ens.T, sr)

        # 2. De-Reverb
        print(f"    [2/4] De-reverberation...")
        dr_dir = os.path.join(raw_dir, "02_DeReverb")
        out_dr = self.engine.separate(ens_path, dr_dir, self.cfg['models']['dereverb'])
        dry_path = self.engine.find_file_by_keyword(out_dr, "No_Reverb", dr_dir)
        
        dry_final_path = os.path.join(raw_dir, '02_Dry_Vocals.wav')
        shutil.copy(dry_path, dry_final_path)

        # 3. Smart Gate
        print(f"    [3/4] Smart Gate...")
        clean_path = os.path.join(raw_dir, '03_Clean_Vocals.wav')
        smart_gate(dry_final_path, clean_path, sr)
        
        # Final Vocal Output
        shutil.copy(clean_path, os.path.join(song_out_dir, '03_vocals_final.wav'))

        # 4. Karaoke Split (Main vs Backing)
        if self.cfg['work_settings']['enable_karaoke']:
            print(f"    [4/4] Main/Backing Split...")
            
            # RoFormer Split
            dir_rof = os.path.join(raw_dir, "04_A_RoF")
            out_rof = self.engine.separate(clean_path, dir_rof, self.cfg['models']['karaoke_rof'])
            main_rof, back_rof = self.engine.identify_stems(out_rof, dir_rof, clean_path)

            # MDX Split
            dir_mdx = os.path.join(raw_dir, "04_B_MDX")
            out_mdx = self.engine.separate(clean_path, dir_mdx, self.cfg['models']['karaoke_mdx'])
            main_mdx, back_mdx = self.engine.identify_stems(out_mdx, dir_mdx, clean_path)

            # Weighted Mix
            w_rof = self.cfg['weights']['karaoke']['RoFormer']
            w_mdx = self.cfg['weights']['karaoke']['MDX']

            main_final, _ = mix_weighted(main_rof, main_mdx, w_rof, w_mdx, sr)
            back_final, _ = mix_weighted(back_rof, back_mdx, w_rof, w_mdx, sr)

            sf.write(os.path.join(song_out_dir, '04_main_vocal.wav'), main_final.T, sr)
            sf.write(os.path.join(song_out_dir, '04_backing_vocal.wav'), back_final.T, sr)
        else:
            print("    [4/4] Skipped.")

        # Cleanup
        if str(self.env.get('SAVE_ALL_ARTIFACTS')).lower() != 'true':
            shutil.rmtree(raw_dir)
            
        return song_out_dir