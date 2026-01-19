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

    def separate(self, input_file, output_dir, model_filename):
        # Initialize Separator
        sep = Separator(
            model_file_dir=self.model_dir,
            output_dir=output_dir,
            output_format=self.temp_format,
            mdx_params=self.params.get('hq_mdx', {}),
            mdxc_params=self.params.get('hq_mdxc', {})
        )
        
        # Load and separate
        sep.load_model(model_filename=model_filename)
        output_files = sep.separate(input_file)
        
        # Cleanup
        del sep
        self.clear_gpu()
        
        return output_files

    def identify_stems(self, file_list, base_dir, reference_src):
        """Find Main vs Backing vocal by comparing with reference"""
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
        found = next((x for x in file_list if keyword in x), file_list[0] if file_list else None)
        return os.path.join(base_dir, found)

    @staticmethod
    def clear_gpu():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()