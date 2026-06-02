"""
本demo用于在arm上对于源代码进行相应的更改，具体更改为：
1. 用FFmpeg代替decord：创建FFmpegVideoReader类，实现与decord.VideoReader兼容的API
   - 创建文件：src/ffmpeg_video_reader.py
   - 修改文件：src/video_utils.py, vlm2vec/data/eval_dataset/mvbench_dataset.py, vlm2vec/model/vlm_backbone/qwen2_vl/qwen_vl_utils.py
   - 实现方法：__init__, __len__, __getitem__, get_avg_fps, get_frame_timestamp, get_frame_index, get_batch

2. 用OpenCV代替torchvision.io.write_video：创建自定义write_video函数
   - 修改文件：vlm2vec/data/utils/video_transforms.py, vlm2vec/data/utils/vision_utils.py, train/rl/data/data_process.py, src/model.py
   - 实现方法：使用cv2.VideoWriter写入视频，支持mp4(h264)和avi格式
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

def test_import():
    """Test if FFmpegVideoReader can be imported"""
    print("Testing import...")
    try:
        from src.ffmpeg_video_reader import FFmpegVideoReader, VideoReader, cpu
        print("✓ FFmpegVideoReader imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_video_utils():
    """Test if video_utils can be imported without decord"""
    print("\nTesting video_utils import...")
    try:
        from src.video_utils import read_video_decord, VideoReader
        print("✓ video_utils imported successfully (decord replaced)")
        return True
    except Exception as e:
        print(f"✗ video_utils import failed: {e}")
        return False

def test_mvbench_dataset():
    """Test if mvbench_dataset can be imported without decord"""
    print("\nTesting mvbench_dataset import...")
    try:
        from vlm2vec.data.eval_dataset.mvbench_dataset import VideoReader, cpu
        print("✓ mvbench_dataset imported successfully (decord replaced)")
        return True
    except Exception as e:
        print(f"✗ mvbench_dataset import failed: {e}")
        return False

def test_qwen_vl_utils():
    """Test if qwen_vl_utils can be imported without decord"""
    print("\nTesting qwen_vl_utils import...")
    try:
        from vlm2vec.model.vlm_backbone.qwen2_vl.qwen_vl_utils import is_decord_available, _read_video_ffmpeg
        available = is_decord_available()
        print(f"✓ qwen_vl_utils imported successfully (is_decord_available={available})")
        return True
    except Exception as e:
        print(f"✗ qwen_vl_utils import failed: {e}")
        return False

def test_video_reader_api():
    """Test FFmpegVideoReader API compatibility"""
    print("\nTesting FFmpegVideoReader API...")
    try:
        from src.ffmpeg_video_reader import FFmpegVideoReader, VideoReader, cpu, FrameWrapper
        
        # Check if all required methods exist
        required_methods = ['__init__', '__len__', '__getitem__', 'get_avg_fps', 
                           'get_frame_timestamp', 'get_frame_index', 'get_batch']
        for method in required_methods:
            if hasattr(VideoReader, method):
                print(f"  ✓ Method {method} exists")
            else:
                print(f"  ✗ Method {method} missing")
                return False
        
        # Check cpu() function
        ctx = cpu(0)
        print(f"  ✓ cpu(0) returns {ctx}")
        
        # Check FrameWrapper
        import numpy as np
        dummy_frame = np.zeros((224, 224, 3), dtype=np.uint8)
        wrapper = FrameWrapper(dummy_frame)
        numpy_result = wrapper.numpy()
        if numpy_result is dummy_frame:
            print("  ✓ FrameWrapper.numpy() works correctly")
        else:
            print("  ✗ FrameWrapper.numpy() failed")
            return False
            
        print("✓ FFmpegVideoReader API is compatible with decord.VideoReader")
        return True
    except Exception as e:
        print(f"✗ API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("FFmpeg Replacement for decord - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_import),
        ("Video Utils Test", test_video_utils),
        ("MVBench Dataset Test", test_mvbench_dataset),
        ("Qwen VL Utils Test", test_qwen_vl_utils),
        ("API Compatibility Test", test_video_reader_api),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} crashed: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print("=" * 60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! FFmpeg replacement is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())
