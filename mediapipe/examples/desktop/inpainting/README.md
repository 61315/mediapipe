# Inpainting example

## Schematics

```
┌─────────────────┐                                           
│IMAGE:input_video│                                           
└┬────────────────┘                                           
┌▽──────────────────────────────────────────┐                 
│IMAGE:throttled_input_video                │                 
└┬───────────────────┬─────────────────────┬┘                 
┌▽─────────────────┐┌▽───────────────────┐┌▽─────────────────┐
│[FaceLandmark]    ││[SelfieSegmentation]││IMAGE:output_video│
└┬────────────────┬┘└────────────────┬───┘└──────────────────┘
┌▽──────────────┐┌▽────────────────┐┌▽────────────────┐       
│IMAGE:face_mask││IMAGE:corpus_mask││IMAGE:selfie_mask│       
└───────────────┘└─────────────────┘└─────────────────┘       
```

## Build and run

```
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/inpainting:inpainting_cpu
```

```
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/inpainting/inpainting_cpu --calculator_graph_config_file=mediapipe/graphs/inpainting/inpainting_cpu.pbtxt
```

Optional `--output_video_path=mediapipe/examples/desktop/inpainting/out.mp4` for video export.

## Visual checkpoint

![video](./out.mp4)