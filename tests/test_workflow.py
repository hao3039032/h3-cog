from h3_workflow import build_raylight_workflow, build_workflow
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
    assert graph["14"]["inputs"]["codec"] == "h264"
    assert graph["14"]["inputs"]["codec.encoding"] == "re-encode"
    assert graph["14"]["inputs"]["codec.encoding.crf"] == 17.0

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
    assert graph["5"]["inputs"]["ref_images.ref_image_1"] == ["15", 0]
    assert graph["5"]["inputs"]["ref_videos.ref_video_1"] == ["17", 0]
    assert graph["5"]["inputs"]["ref_video_audios.ref_video_audio_1"] == ["17", 1]
    assert graph["18"] == {"class_type": "LoadAudio", "inputs": {"audio": "voice.wav"}}
    assert graph["5"]["inputs"]["ref_audios.ref_audio_1"] == ["18", 0]


def test_raylight_workflow_shards_h3_across_two_4090s():
    graph = build_raylight_workflow(
        prompt="Use <Picture 1>.", width=480, height=864, frames=124,
        steps=20, seed=42, reference_image_names=["identity.png"],
    )
    initializer = graph["1"]
    assert initializer["class_type"] == "RayInitializer"
    assert initializer["inputs"]["GPU"] == 2
    assert initializer["inputs"]["ulysses_degree"] == 2
    assert initializer["inputs"]["ring_degree"] == 1
    assert initializer["inputs"]["FSDP"] is True
    assert initializer["inputs"]["FSDP_CPU_OFFLOAD"] is False
    assert initializer["inputs"]["use_mmap"] is True
    assert graph["2"]["class_type"] == "RayUNETLoader"
    assert graph["2"]["inputs"]["unet_name"] == "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
    assert graph["7"] == {"class_type": "RayBasicGuider", "inputs": {"ray_actors": ["2", 0], "conditioning": ["6", 0]}}
    assert graph["9"]["class_type"] == "RayBasicScheduler"
    assert graph["10"]["class_type"] == "XFuserSamplerCustomAdvanced"
    assert graph["10"]["inputs"]["noise_seed"] == 42
    assert graph["15"] == {"class_type": "LoadImage", "inputs": {"image": "identity.png"}}
    assert graph["6"]["inputs"]["ref_images.ref_image_1"] == ["15", 0]


def test_raylight_easycache_uses_distributed_patch_node():
    cache = CacheTuning("balanced", 0.12, 0.15, 0.9, False, "sweep-1", "balanced")
    graph = build_raylight_workflow(
        prompt="p", width=480, height=864, frames=124, steps=20, seed=9,
        reference_image_names=["identity.png"], cache=cache,
    )
    assert graph["15"]["class_type"] == "RayEasyCache"
    assert graph["15"]["inputs"]["ray_actors"] == ["2", 0]
    assert graph["15"]["inputs"]["distributed_sync"] is True
    assert graph["7"]["inputs"]["ray_actors"] == ["15", 0]
    assert graph["9"]["inputs"]["ray_actors"] == ["15", 0]
