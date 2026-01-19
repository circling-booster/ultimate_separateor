import os
import gc
import torch
import librosa
import numpy as np
from audio_separator.separator import Separator

class InferenceEngine:
    def __init__(self, model_dir, temp_format='wav', params=None):
        self.model_dir = model_dir
        self.temp_format = temp_format
        self.params = params if params else {}

    def separate(self, input_file, output_dir, model_filename, **kwargs):
        """
        Runs separation with optional parameter overrides.
        kwargs can contain 'mdx_params' or 'mdxc_params' dictionaries to override defaults.
        """
        # 기본 설정 가져오기 (Deep Copy로 원본 보존)
        mdx_params = self.params.get('hq_mdx', {}).copy()
        mdxc_params = self.params.get('hq_mdxc', {}).copy()

        # 특정 모델을 위한 파라미터 덮어쓰기 (예: Reverb 모델용 segment_size=256)
        if 'mdx_params' in kwargs:
            mdx_params.update(kwargs['mdx_params'])
        if 'mdxc_params' in kwargs:
            mdxc_params.update(kwargs['mdxc_params'])

        # Separator 초기화
        sep = Separator(
            model_file_dir=self.model_dir,
            output_dir=output_dir,
            output_format=self.temp_format,
            mdx_params=mdx_params,
            mdxc_params=mdxc_params
        )
        
        print(f"        🔹 Loading model: {model_filename}...")
        try:
            sep.load_model(model_filename=model_filename)
            
            # 분리 실행
            output_files = sep.separate(input_file)
        except Exception as e:
            # 모델 로드나 분리 중 에러 발생 시 명확히 알림
            raise RuntimeError(f"Failed to run model '{model_filename}': {str(e)}")
        finally:
            # 메모리 정리
            del sep
            self.clear_gpu()
        
        if not output_files:
            raise RuntimeError(f"Separation failed for model {model_filename}. No output files were generated.")
            
        return output_files

    def identify_stems(self, file_list, base_dir, reference_src):
        """Find Main vs Backing vocal by comparing with reference"""
        if not file_list:
             raise ValueError("File list is empty. Cannot identify stems.")

        y_ref, _ = librosa.load(reference_src, sr=None, mono=True)
        scores = []
        
        for f in file_list:
            path = os.path.join(base_dir, f)
            y_target, _ = librosa.load(path, sr=None, mono=True)
            min_len = min(len(y_ref), len(y_target))
            # Dot product as similarity score
            score = np.dot(y_ref[:min_len], y_target[:min_len])
            scores.append((f, score))
            
        scores.sort(key=lambda x: x[1], reverse=True)
        # Top score is Main, second is Backing
        main_file = os.path.join(base_dir, scores[0][0])
        back_file = os.path.join(base_dir, scores[1][0] if len(scores) > 1 else scores[0][0])
        return main_file, back_file

    def find_file_by_keyword(self, file_list, keyword, base_dir):
        if not file_list:
            raise ValueError(f"Cannot find keyword '{keyword}': File list is empty.")
            
        found = next((x for x in file_list if keyword in x), None)
        
        if found is None:
            available = ", ".join(file_list)
            raise FileNotFoundError(f"Keyword '{keyword}' not found in output files: [{available}]")

        return os.path.join(base_dir, found)

    @staticmethod
    def clear_gpu():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()