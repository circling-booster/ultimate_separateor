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
        
        # 원본 길이 참조용 로드 (Skip 여부와 관계없이 필요)
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
                
                # 서브 모델 결과가 있는지 확인 (재사용)
                # 주의: 여기서는 Vocals 파일 이름을 추정해야 하므로, 간단히 폴더 내 'Vocals' 키워드 파일 검색
                # 만약 없다면 추론 실행
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

            # Mix Vocals
            print(f"        🔹 Mixing...")
            voc_ens, _ = align_and_mix(stem_paths, self.cfg['weights']['ensemble_2_model']['Balanced'], sr)
            
            # Length Fix
            min_len = min(y_orig.shape[-1], voc_ens.shape[-1])
            voc_ens = voc_ens[..., :min_len]
            sf.write(ens_path, voc_ens.T, sr)

        # ---------------------------------------------------------
        # [Step 2] De-Reverb
        # ---------------------------------------------------------
        dry_final_path = os.path.join(raw_dir, '02_Dry_Vocals.wav')
        
        if os.path.exists(dry_final_path):
            print(f"    [2/4] ✅ Found dry vocals. Skipping...")
        else:
            print(f"    [2/4] De-reverberation...")
            dr_dir = os.path.join(raw_dir, "02_DeReverb")
            
            # 중간 단계(DeReverb 결과) 확인
            out_dr_files = []
            if os.path.exists(dr_dir):
                 out_dr_files = [f for f in os.listdir(dr_dir) if "No_Reverb" in f]
            
            if out_dr_files:
                 dry_path = os.path.join(dr_dir, out_dr_files[0])
            else:
                 out_dr = self.engine.separate(ens_path, dr_dir, self.cfg['models']['dereverb'])
                 dry_path = self.engine.find_file_by_keyword(out_dr, "No_Reverb", dr_dir)
            
            shutil.copy(dry_path, dry_final_path)

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
        # [Step 4] Karaoke Split (Main vs Backing)
        # ---------------------------------------------------------
        main_out = os.path.join(song_out_dir, '04_main_vocal.wav')
        back_out = os.path.join(song_out_dir, '04_backing_vocal.wav')
        
        if self.cfg['work_settings']['enable_karaoke']:
            if os.path.exists(main_out) and os.path.exists(back_out):
                print(f"    [4/4] ✅ Found karaoke split. Skipping...")
            else:
                print(f"    [4/4] Main/Backing Split...")
                
                # RoFormer Split
                dir_rof = os.path.join(raw_dir, "04_A_RoF")
                main_rof, back_rof = self._get_or_run_karaoke(
                    clean_path, dir_rof, self.cfg['models']['karaoke_rof']
                )

                # MDX Split
                dir_mdx = os.path.join(raw_dir, "04_B_MDX")
                main_mdx, back_mdx = self._get_or_run_karaoke(
                    clean_path, dir_mdx, self.cfg['models']['karaoke_mdx']
                )

                # Weighted Mix
                print(f"        🔹 Mixing Karaoke Stems...")
                w_rof = self.cfg['weights']['karaoke']['RoFormer']
                w_mdx = self.cfg['weights']['karaoke']['MDX']

                main_final, _ = mix_weighted(main_rof, main_mdx, w_rof, w_mdx, sr)
                back_final, _ = mix_weighted(back_rof, back_mdx, w_rof, w_mdx, sr)

                sf.write(main_out, main_final.T, sr)
                sf.write(back_out, back_final.T, sr)
        else:
            print("    [4/4] Skipped (Disabled).")

        # Cleanup (Optional: Only if completely finished and saving artifacts is disabled)
        if str(self.env.get('SAVE_ALL_ARTIFACTS')).lower() != 'true':
            # 주의: Resume 기능을 위해 삭제 로직은 신중해야 함.
            # 작업이 완료되었을 때만 삭제하거나, 사용자가 명시적으로 원할 때만 삭제.
            # 여기서는 안전을 위해 주석 처리하거나, 필요 시 활성화.
            # shutil.rmtree(raw_dir)
            pass
            
        return song_out_dir

    def _get_or_run_karaoke(self, input_path, work_dir, model_file):
        """Helper to check if karaoke split files already exist"""
        # 폴더 내에 2개의 파일이 이미 존재하면 그것을 사용 (파일명 정렬로 추정)
        # 하지만 정확성을 위해 identify_stems 로직을 다시 태우는 것이 안전함.
        # 분리 자체가 시간 소요가 크므로, 분리 파일이 있으면 로드만 함.
        
        existing_files = []
        if os.path.exists(work_dir):
            existing_files = [os.path.join(work_dir, f) for f in os.listdir(work_dir) if f.endswith('.wav')]
        
        if len(existing_files) >= 2:
            print(f"        🔹 Using cached output from {os.path.basename(work_dir)}...")
            # 이미 파일이 있다면 Main/Back 구분 로직만 다시 실행 (매우 빠름)
            return self.engine.identify_stems([os.path.basename(f) for f in existing_files], work_dir, input_path)
        else:
            print(f"        🔹 Running {os.path.basename(work_dir)}...")
            out_files = self.engine.separate(input_path, work_dir, model_file)
            return self.engine.identify_stems(out_files, work_dir, input_path)