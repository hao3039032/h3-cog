from h3_workflow import build_workflow
from h3_tuning import CacheTuning


def test_ref2va_workflow_has_official_h3_sampling_path():
    graph = build_workflow(
        prompt="p", width=1024, height=576, frames=124, steps=20, seed=7,
        reference_image_names=["identity.png"],
    )
    assert graph["1"]["inputs"]["unet_name"] == "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
    assert graph["2"]["inputs"]["type"] == "minimax"
    assert graph["5"]["class_type"] == "MiniMaxH3ReferenceToVideo"
    assert graph["8"]["inputs"]["sampler_name"] == "res_multistep"
    assert graph["9"]["inputs"] == {"model": ["1", 0], "scheduler": "simple", "steps": 20, "denoise": 1.0}
    assert graph["13"]["inputs"]["audio"] == ["12", 0]

def test_easycache_is_only_inserted_when_operator_tuning_is_supplied():
    cache = CacheTuning("balanced", 0.12, 0.15, 0.9, False, "sweep-1", "balanced")
    graph = build_workflow(
        prompt="p", width=768, height=768, frames=124, steps=20, seed=9,
        reference_image_names=["identity.png"], cache=cache,
    )
    assert graph["15"]["class_type"] == "EasyCache"
    assert graph["15"]["inputs"]["model"] == ["1", 0]
    assert graph["6"]["inputs"]["model"] == ["15", 0]
    assert graph["9"]["inputs"]["model"] == ["15", 0]
    assert graph["16"] == {"class_type": "LoadImage", "inputs": {"image": "identity.png"}}


def test_reference_workflow_uses_ref2va_and_native_media_loaders():
    graph = build_workflow(
        prompt="Use <Picture 1>, <Video 1>, and <Audio 2>.",
        width=864,
        height=480,
        frames=124,
        steps=20,
        seed=11,
        reference_image_names=["identity.png"],
        reference_video_names=["motion.mp4"],
        reference_audio_names=["voice.wav"],
    )
    assert graph["1"]["inputs"]["unet_name"] == "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
    assert graph["5"]["class_type"] == "MiniMaxH3ReferenceToVideo"
    assert graph["5"]["inputs"]["audio_vae"] == ["4", 0]
    assert graph["15"] == {"class_type": "LoadImage", "inputs": {"image": "identity.png"}}
    assert graph["16"] == {"class_type": "LoadVideo", "inputs": {"file": "motion.mp4"}}
    assert graph["17"] == {"class_type": "GetVideoComponents", "inputs": {"video": ["16", 0]}}
    assert graph["5"]["inputs"]["ref_video_1"] == ["17", 0]
    assert graph["5"]["inputs"]["ref_video_audio_1"] == ["17", 1]
    assert graph["18"] == {"class_type": "LoadAudio", "inputs": {"audio": "voice.wav"}}
