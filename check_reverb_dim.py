import onnxruntime as ort
import os

model_path = "./models/Reverb_HQ_By_FoxJoy.onnx"

print("="*40)
print(f"🔍 Inspecting: {os.path.basename(model_path)}")
print("="*40)

if not os.path.exists(model_path):
    print("❌ Model file not found!")
    exit(1)

try:
    # ONNX 모델 로드
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    inputs = sess.get_inputs()
    
    for i in inputs:
        print(f"🔹 Input Name: {i.name}")
        print(f"🔹 Shape: {i.shape}")
        
        # MDX 모델의 dim_t는 보통 마지막 차원입니다.
        if len(i.shape) == 4:
            dim_t = i.shape[-1]
            print(f"\n✅ Detected dim_t (Segment Size): {dim_t}")
            if isinstance(dim_t, str):
                print("⚠️  Dynamic dimension detected. Try default 256.")
        else:
            print(f"⚠️  Unexpected shape format.")

except Exception as e:
    print(f"❌ Error inspecting model: {e}")

print("="*40)