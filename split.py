import os
import logging
from pathlib import Path
from audio_separator.separator import Separator
import soundfile as sf
import librosa
import numpy as np

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ===== 설정 =====
VOCALS_DIR = Path("vocals")       # chiller.wav, sayhello.wav, candy.wav가 있는 폴더
OUTPUT_DIR = Path("output_split") # 메인/백 보컬이 저장될 폴더

# UVR VR(Karaoke) 모델 파일명
VR_MODEL_FILENAME = "5_HP-Karaoke-UVR.pth"

# 곡별 VR 파라미터 튜닝 값
TRACK_CONFIGS = {
    "chiller.wav": {
        "description": "보이그룹, 랩 많음, 본인 화음 많음 → 공격적인 메인 추출",
        "vr_params": {
            "batch_size": 1,
            "window_size": 320,
            "aggression": 8,
            "enable_tta": True,
            "enable_post_process": True,
            "post_process_threshold": 0.15,
            "high_end_process": False,
        }
    },
    "sayhello.wav": {
        "description": "여성 솔로, 본인 화음+코러스, 느린 템포 → 균형잡힌 분리",
        "vr_params": {
            "batch_size": 1,
            "window_size": 320,
            "aggression": 5,
            "enable_tta": True,
            "enable_post_process": False,
            "post_process_threshold": 0.2,
            "high_end_process": False,
        }
    },
    "candy.wav": {
        "description": "보이그룹, 중간 템포, 멤버 합창 있음 → 합창 고려한 분리",
        "vr_params": {
            "batch_size": 1,
            "window_size": 320,
            "aggression": 6,
            "enable_tta": True,
            "enable_post_process": True,
            "post_process_threshold": 0.2,
            "high_end_process": True,
        }
    },
}


def ensure_mono_44100(input_path: Path, tmp_dir: Path) -> Path:
    """
    오디오 파일을 44.1kHz 스테레오로 정규화
    """
    try:
        y, sr = librosa.load(input_path, sr=44100, mono=False)
    except Exception as e:
        logger.warning(f"Librosa 로드 실패 ({input_path}): {e}, soundfile 사용")
        data, sr = sf.read(input_path)
        y = data.T if data.ndim == 2 else data
        if sr != 44100:
            y = librosa.resample(y if y.ndim == 1 else y[0], orig_sr=sr, target_sr=44100)

    # 모노 → 스테레오 변환
    if y.ndim == 1:
        y = np.stack([y, y], axis=0)
    elif y.shape[0] == 1:
        y = np.repeat(y, 2, axis=0)

    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_path = tmp_dir / (input_path.stem + "_44k_stereo.wav")
    
    # 스테레오 (2, N) → (N, 2) 변환 후 저장
    if y.shape[0] == 2:
        y = y.T
    
    sf.write(out_path, y, 44100)
    logger.info(f"  정규화 완료: {out_path}")
    return out_path


def split_main_back_for_track(
    track_name: str,
    vocals_dir: Path,
    output_dir: Path,
    vr_model_filename: str,
    track_config: dict,
):
    """
    단일 트랙에 대해 VR(Karaoke) 모델을 이용해 메인/백 보컬을 분리합니다.
    
    입력: 보컬 스템 (이미 반주 제거된 상태)
    출력:
        output_dir/track_stem/main.wav      (메인/리드 보컬)
        output_dir/track_stem/back.wav      (백 보컬/코러스)
    """
    input_path = vocals_dir / track_name
    if not input_path.exists():
        logger.warning(f"[ERROR] {input_path}가 존재하지 않습니다. 건너뜀.")
        return

    logger.info(f"\n{'='*70}")
    logger.info(f"[곡 분석] {track_name}")
    logger.info(f"설명: {track_config['description']}")
    logger.info(f"{'='*70}")

    track_out_dir = output_dir / input_path.stem
    track_out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 샘플레이트/채널 정규화 (44.1kHz / 스테레오)
    logger.info(f"\n[1단계] 오디오 정규화 (44.1kHz, 스테레오)...")
    tmp_dir = track_out_dir / "_tmp"
    normalized_input = ensure_mono_44100(input_path, tmp_dir)

    # 2) Separator 인스턴스 생성 (곡별 VR 파라미터 적용)
    logger.info(f"\n[2단계] Separator 초기화 (VR 모델)...")
    separator = Separator(
        output_dir=str(track_out_dir),
        output_format="WAV",
        use_autocast=True,  # GPU 가속 활성화
        vr_params=track_config["vr_params"],
    )

    # 3) VR(Karaoke) 모델 로드
    logger.info(f"\n[3단계] VR 모델 로드 ({vr_model_filename})...")
    try:
        separator.load_model(model_filename=vr_model_filename)
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")
        return

    # 4) 분리 실행
    logger.info(f"\n[4단계] 메인/백 보컬 분리 실행...")
    logger.info(f"  VR 파라미터:")
    for key, val in track_config["vr_params"].items():
        logger.info(f"    - {key}: {val}")
    
    # output_names는 딕셔너리로 전달
    output_names = {
        "Vocals": "main",      # VR 모델의 Vocals 출력 → main.wav
        "Instrumental": "back"  # VR 모델의 Instrumental 출력 → back.wav
    }

    try:
        output_files = separator.separate(str(normalized_input), output_names=output_names)
        logger.info(f"  분리 완료! 생성 파일:")
        for file in output_files:
            logger.info(f"    ✓ {Path(file).name}")
    except Exception as e:
        logger.error(f"분리 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5) 결과 파일 확인 및 로깅
    logger.info(f"\n[5단계] 결과 파일 확인...")
    main_path = track_out_dir / "main.wav"
    back_path = track_out_dir / "back.wav"

    if main_path.exists():
        main_duration = len(librosa.get_samplerate(str(main_path))[1]) / librosa.get_samplerate(str(main_path))[0]
        logger.info(f"  ✓ 메인 보컬: {main_path.name} ({main_duration:.2f}s)")
    
    if back_path.exists():
        back_duration = len(librosa.get_samplerate(str(back_path))[1]) / librosa.get_samplerate(str(back_path))[0]
        logger.info(f"  ✓ 백 보컬:  {back_path.name} ({back_duration:.2f}s)")

    # 6) 임시 파일 정리
    try:
        for f in tmp_dir.glob("*"):
            f.unlink()
        tmp_dir.rmdir()
    except Exception:
        pass

    logger.info(f"\n[완료] {track_name} → {track_out_dir}")


def main():
    """메인 실행 함수"""
    logger.info("\n" + "="*70)
    logger.info("한국 보컬 음원 메인/백 보컬 분리 시스템 v1.0")
    logger.info("모델: UVR VR(Karaoke) Architecture - 5_HP-Karaoke-UVR.pth")
    logger.info("="*70)

    # 입력/출력 디렉토리 확인
    if not VOCALS_DIR.exists():
        logger.error(f"[ERROR] {VOCALS_DIR} 디렉토리가 없습니다.")
        logger.error(f"  {VOCALS_DIR.absolute()} 에 다음 파일들을 배치하세요:")
        for track_name in TRACK_CONFIGS.keys():
            logger.error(f"    - {track_name}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 각 곡별로 분리 실행
    for track_name, cfg in TRACK_CONFIGS.items():
        split_main_back_for_track(
            track_name=track_name,
            vocals_dir=VOCALS_DIR,
            output_dir=OUTPUT_DIR,
            vr_model_filename=VR_MODEL_FILENAME,
            track_config=cfg,
        )

    # 최종 결과 요약
    logger.info("\n" + "="*70)
    logger.info("[최종 결과 요약]")
    logger.info("="*70)

    for track_name in TRACK_CONFIGS.keys():
        track_out_dir = OUTPUT_DIR / Path(track_name).stem
        main_path = track_out_dir / "main.wav"
        back_path = track_out_dir / "back.wav"
        
        logger.info(f"\n📁 {track_name}")
        if main_path.exists() and back_path.exists():
            logger.info(f"  ✅ 분리 완료")
            logger.info(f"    - 메인 보컬: {main_path}")
            logger.info(f"    - 백 보컬:   {back_path}")
        else:
            logger.info(f"  ⚠️  분리 실패 (파일 생성 안됨)")

    logger.info(f"\n📁 전체 출력 디렉토리: {OUTPUT_DIR.absolute()}")
    logger.info("="*70)


if __name__ == "__main__":
    main()