import os
import cv2
import glob
try:
    from openvino.inference_engine import IECore
    from accuracy_checker.preprocessor.ie_preprocessor import IEPreprocessor
except Exception as e:
    print(e)


def get_image(path, dst_size, color='bgr', flag='IR'):
    """
    :param color: bgr or rgb
    :param dst_size:
    :param path:
    :return:
    """
    if flag == 'IR':
        image = cv2.imread(path)
        return transform_fn(image, dst_size, color=color)
    else:
        raise NotImplementedError


def transform_fn(image, dst_size=None, color='bgr'):
    if color is 'rgb':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, dst_size, cv2.INTER_LINEAR)
    x = image.transpose(2, 0, 1)
    return x


# def IE_bilinear_resize():
#     config = [{'type': 'resize', 'interpolation': 'bilinear'}]
#     preprocessor = IEPreprocessor(config)


def glob_one_files(pattern):
    files = glob.glob(pattern)
    if len(files) == 1:
        return files[0]
    else:
        if len(files) == 0:
            print('None file exists in {}'.format(pattern))
        else:
            print('Multiple files are found: {}'.format(files))
        raise ValueError


def get_IE_net(model_dir, executable_model=None, net=None):
    """
        only load inference engine model, do not process
        At first call, executable_model and net can be None,
            for efficiently, you can save these param and send in after first called.
    :param model_dir:
    :param executable_model:
    :param net:
    :return:
    """
    # Load the model.
    device = "CPU"
    if executable_model is None or net is None:
        if os.path.isdir(model_dir):
            ir_model_path = glob_one_files(os.path.join(model_dir, '*.xml'))
            ir_weights_path = glob_one_files(os.path.join(model_dir, '*.bin'))
        else:
            ir_model_path = model_dir  # model.xml
            ir_weights_path = model_dir.replace('.xml', '.bin')
        print('loading IR model:{}'.format(ir_model_path))

        # inference engine
        ie = IECore()
        # read IR
        net = ie.read_network(model=ir_model_path, weights=ir_weights_path)
        # load model
        executable_model = ie.load_network(network=net, device_name=device)
        print('loaded IR model done.')
    return executable_model, net


def get_IE_output(model_dir, image, executable_model=None, net=None):
    """
        only support single input_blob
    :param model_dir:
    :param image:
    :param executable_model:
    :param net:
    :return:
    """
    # Load the model.
    executable_model, net = get_IE_net(model_dir, executable_model, net)

    # get input and output name
    assert len(net.input_info) == 1, 'not support: {} input blobs'.format(len(net.input_info))
    input_blob = next(iter(net.input_info))

    # assert len(net.outputs) == 1, 'not support: {} output blobs'.format(len(net.outputs))
    # output_blob = next(iter(net.outputs))

    # start sync inference
    predictions = executable_model.infer(inputs={input_blob: image})
    return executable_model, net, predictions
