# from importlib import util
from pydoc import locate
import inspect
import argparse
import numpy as np
from PIL import Image

def argparse_summary(arg_list, parser):
    arg_dict = vars(arg_list)
    action_groups_dict = {}
    for i in range(len(parser._action_groups)):
        action_groups_dict[parser._action_groups[i].title]=[]
    for j in parser._actions:
        if j.dest == "help":
            continue
        try:
            action_groups_dict[j.container.title].append((j.dest, arg_dict[j.dest]))
        except:
            print(f"not working: {j.dest}")

    value = "########################ArgParseSummaryStart########################"
    len_group_var = 55
    for k in parser._action_groups:
        group = k.title
        length_filler = len_group_var-len(group)
        length_filler1 = length_filler-(length_filler//2)
        length_filler2 = length_filler-length_filler1
        value+= f"\n{''.join(['-']*length_filler1)}{group}{''.join(['-']*length_filler2)}"
        for l in action_groups_dict[group]:
            value += "\n  {0:<25s}: {1:21s}  ".format(l[0], str(l[1]))
    value += "\n########################ArgParseSummaryEnd########################"
    print(value)


def get_argparser_group(title, parser):
    for group in parser._action_groups:
        if title == group.title:
            return group
    return None


def get_class_by_path(dot_path=None):
    if dot_path:
        MyClass = locate(dot_path)
        assert inspect.isclass(MyClass), f"Could not load {dot_path}"
        return MyClass
    else:
        return None


def get_function_by_path(dot_path=None):
    if dot_path:
        myfunction = locate(dot_path)
        assert inspect.isfunction(myfunction), f"Could not load {dot_path}"
        return myfunction
    else:
        return None


def get_model_by_function_path(hparams):
    model_constructor = get_function_by_path("models." + hparams.model)
    model = model_constructor(hparams)
    return model


def get_model_by_class_path(hparams):
    ModelClass = get_class_by_path("models." + hparams.model)
    model = ModelClass(hparams)
    return model


def get_dataset_by_class_path(hparams):
    DatasetClass = get_class_by_path("datasets." + hparams.dataset)
    dataset = DatasetClass(hparams)
    return dataset

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def tensor2np_array(input_tensor):
    np_array_batch = np.squeeze(input_tensor.numpy())

    output_data = []
    for i in range(np_array_batch.shape[0]):
        np_array = np.squeeze(np_array_batch[i, ...])
        output_data.append(np_array)

    return output_data

def save_data(data, filename, fmt='npy', is_label=False):
    if fmt == 'npy':
        np.save(filename + "." + fmt, data)

    elif fmt == 'png':

        rescaled_image = data - np.min(data)
        rescaled_image = rescaled_image / np.max(rescaled_image) * 255
        rescaled_image = rescaled_image.astype(np.uint8)

        if is_label:
            rescaled_image = np.where(rescaled_image > 0, 1, 0)

        pil_image = Image.fromarray(rescaled_image)
        pil_image.save(filename + "." + fmt)


