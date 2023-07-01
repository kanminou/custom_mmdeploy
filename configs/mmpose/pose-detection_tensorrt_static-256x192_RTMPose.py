_base_ = ['./pose-detection_static.py', '../_base_/backends/tensorrt.py']

onnx_config = dict(input_shape=[192, 256],output_names=['simcc_x','simcc_y'])
backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 256, 192],
                    opt_shape=[1, 3, 256, 192],
                    max_shape=[1, 3, 256, 192])))
    ])