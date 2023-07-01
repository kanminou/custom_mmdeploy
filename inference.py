from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
import torch
import time

# deploy_cfg = 'configs/mmpose/pose-detection_tensorrt_static-256x192.py'
# model_cfg = '../mmpose/configs/body_2d_keypoint/topdown_regression/coco/td-reg_fastvgg_s_rle-8xb64-210e_coco-256x192.py'
# device = 'cuda'
# backend_model = ['mmdeploy_models/mmpose/fastvgg_s_trt/end2end.engine']
# image = '000033016.jpg'

deploy_cfg = 'configs/mmpose/pose-detection_ncnn_static-256x192_RTMPose.py'
deploy_cfg = 'configs/mmpose/pose-detection_ncnn_static-256x192_RTMPose_int8.py'
model_cfg = '../mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-t_8xb256-420e_coco-256x192.py'
device = 'cpu'
backend_model = ['mmdeploy_models/mmpose/rtmpose_t_ncnn_int8/end2end_int8.param','mmdeploy_models/mmpose/rtmpose_t_ncnn_int8/end2end_int8.bin']
# backend_model = ['mmdeploy_models/mmpose/rtmpose_t/end2end.onnx']
image = '000033016.jpg'

# read deploy_cfg and model_cfg
deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

# build task and backend model
task_processor = build_task_processor(model_cfg, deploy_cfg, device)
model = task_processor.build_backend_model(backend_model)

# process input image
input_shape = get_input_shape(deploy_cfg)
model_inputs, _ = task_processor.create_input(image, input_shape)

# do model inference
num_warmup = 5
pure_inf_time = 0

    # benchmark with total batch and take the average
for i in range(1000):

    start_time = time.perf_counter()
    with torch.no_grad():
        result = model.test_step(model_inputs)
    elapsed = time.perf_counter() - start_time

    if i >= num_warmup:
        pure_inf_time += elapsed
        if (i + 1) % 100 == 0:
            its = (i + 1 - num_warmup)/ pure_inf_time
            print(f'Done item [{i + 1:<3}],  {its:.2f} items / s')
print(f'Overall average: {its:.2f} items / s')
print(f'Total time: {pure_inf_time:.2f} s')

      

# visualize results
task_processor.visualize(
    image=image,
    model=model,
    result=result[0],
    window_name='visualize',
    output_file='output_pose.png')