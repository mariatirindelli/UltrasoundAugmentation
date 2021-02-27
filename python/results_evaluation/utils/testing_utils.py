import os
import yaml
import itertools

def get_different_keys(dict_list, exclude_keys=()):
    """
    Fing keys corresponding to values that changes among the different dicts in dict_list.
    Returns a dict like:
    different_keys_values = {'params1':[value1, value2, ..],
                            'params2':[value6, value7, ..],
                             ...}

    where params1, params2, .. are the keys which values changes among dictionaries, and value1, value2, .. the values
    the paramters have in the different dicts.
    """

    ref_dic = dict_list[0]
    assert isinstance(ref_dic, dict)

    different_keys_values = dict()
    for key in ref_dic.keys():
        if key in exclude_keys:
            continue

        key_values = [current_dict[key] for current_dict in dict_list]
        is_key_value_equal4all = all(x == key_values[0] for x in key_values)
        if not is_key_value_equal4all:

            params_values = list(set(key_values))
            different_keys_values[key] = params_values

    return different_keys_values

def read_yaml(yaml_path):
    with open(yaml_path, 'r') as fid:
        lines = fid.readlines()

    with open("tmp.yaml", 'w') as fid:
        for line in lines:
            if "output_path" in line or line[0] == "-":
                continue
            fid.write(line + "\n")

    with open("tmp.yaml", 'r') as stream:
        current_config_dict = yaml.load(stream, Loader=yaml.Loader)

    return current_config_dict

def get_dicts_and_varying_params(group_path):

    """
    Returns the list of parameters which differ amongs different config ("output_path", "data_root" keys are not
    considered).

    e.g.

    group/exp1/.../hparams.yml   group/exp2/.../hparams.yml    group/exp3/.../hparams.yml   group/exp3/.../hparams.yml

    data_root: /cross_val0       data_root: cross_val0         data_root: cross_val0        data_root: cross_val0
    loss_function: BCE+Dice      loss_function: BCE+Dice       loss_function: BCE           loss_function: BCE
    use_positive_weights: true   use_positive_weights: false   use_positive_weights: true   use_positive_weights: false
    val_percent_check: 1.0       val_percent_check: 1.0        val_percent_check: 1.0       val_percent_check: 1.0
    num_workers: 6               num_workers: 6                num_workers: 6               num_workers: 6
    on_polyaxon: true            on_polyaxon: true             on_polyaxon: true            on_polyaxon: true
    out_channels: 1              out_channels: 1               out_channels: 1              out_channels: 1
    ...                             ...                             ....                        ...

    It will return dict({loss_function : [BCE+Dice, BCE], use_positive_weights : [true, false]}),
        dict{expId1: dict1, expId2: dict2, expId3: dict3, expId4: dict4, ..}

    different_params = {loss_function : [BCE+Dice, BCE], use_positive_weights : [true, false]}

    cross_folder_dict ={
                        50206 : {
                                    data_root: /data/BoneSegmentation/linear_probe/cross_val0
                                    ...
                                    loss_function: BCE+Dice
                                    max_epochs: 1
                                    use_positive_weights: true
                                }
                        50207 : {
                                    data_root: /data/BoneSegmentation/linear_probe/cross_val0
                                    ...
                                    loss_function: BCE+Dice
                                    max_epochs: 1
                                    use_positive_weights: false
                                }
                        50208 : {
                                    data_root: /data/BoneSegmentation/linear_probe/cross_val0
                                    ...
                                    loss_function: BCE
                                    max_epochs: 1
                                    use_positive_weights: true
                                }
                                    ...
                        }

    """

    experiment_config_dict = dict()
    for experiment in os.listdir(group_path):
        yml_config_path = os.path.join(group_path, experiment, "tb", "version_0", "hparams.yaml")
        current_config_dict = read_yaml(yml_config_path)
        experiment_config_dict[experiment] = current_config_dict

    dict_list = [experiment_config_dict[item] for item in experiment_config_dict.keys()]
    different_params = get_different_keys(dict_list, exclude_keys=["output_path", "data_root", "name"])

    return different_params, experiment_config_dict

def get_params_combinations(param_list):

    combinations = list(itertools.product(*param_list))
    return combinations

def get_crossval_folders_ids(experiment_config_dict, params_combination):
    """
    dict_list = [
    dict1                        dict2                         dict3                        dict4
    {                            {                             {                            {
    data_root: /cross_val0       data_root: cross_val0         data_root: cross_val0        data_root: cross_val0
    loss_function: BCE+Dice      loss_function: BCE+Dice       loss_function: BCE           loss_function: BCE
    use_positive_weights: true   use_positive_weights: false   use_positive_weights: true   use_positive_weights: false
    val_percent_check: 1.0       val_percent_check: 1.0        val_percent_check: 1.0       val_percent_check: 1.0
    num_workers: 6               num_workers: 6                num_workers: 6               num_workers: 6
    on_polyaxon: true            on_polyaxon: true             on_polyaxon: true            on_polyaxon: true
    out_channels: 1              out_channels: 1               out_channels: 1              out_channels: 1
    ...                             ...                             ....                        ...
    }                            }                              }                           } ]

    params_combination[0] = {loss_function : BCE+Dice,use_positive_weights : true}
    params_combination[1] = {loss_function : BCE+Dice,use_positive_weights : false}
    params_combination[3] = {loss_function : BCE,use_positive_weights : true}
    params_combination[4] = {loss_function : BCE,use_positive_weights : true}

    """

    cross_val_folder_list = []
    for experimentId in experiment_config_dict:
        config = experiment_config_dict[experimentId]

        include_config = all(config[param] == params_combination[param] for param in params_combination.keys())
        if include_config:
            cross_val_folder_list.append(experimentId)

    return cross_val_folder_list

def get_cross_val_groups(group_path):
    """
    Returns the list of parameters which differ amongs different config ("output_path", "data_root" keys are not
    considered).

    e.g.
    evaluated_params = dict({loss_function : [BCE+Dice, BCE], use_positive_weights : [true, false]})
    dict_list = [
    dict1                        dict2                         dict3                        dict4
    {                            {                             {                            {
    data_root: /cross_val0       data_root: cross_val0         data_root: cross_val0        data_root: cross_val0
    loss_function: BCE+Dice      loss_function: BCE+Dice       loss_function: BCE           loss_function: BCE
    use_positive_weights: true   use_positive_weights: false   use_positive_weights: true   use_positive_weights: false
    val_percent_check: 1.0       val_percent_check: 1.0        val_percent_check: 1.0       val_percent_check: 1.0
    num_workers: 6               num_workers: 6                num_workers: 6               num_workers: 6
    on_polyaxon: true            on_polyaxon: true             on_polyaxon: true            on_polyaxon: true
    out_channels: 1              out_channels: 1               out_channels: 1              out_channels: 1
    ...                             ...                             ....                        ...
    }                            }                              }                           } ]

    combinations = ((BCE+Dice, true), (BCE+Dice, false), (BCE, true), (BCE, false) )


    cross_folder_dict ={CrossValGroup0 : {
                                        loss_function : BCE+Dice,
                                        use_positive_weights : true
                                        cross_val_folders_ids : [..]
                                        }
                        CrossValGroup1 = {
                                        loss_function : BCE+Dice,
                                        use_positive_weights : false
                                        cross_val_folders_ids : [..]
                                        }
                        CrossValGroup2 = {
                                        loss_function : BCE,
                                        use_positive_weights : true
                                        cross_val_folders_ids : [..]
                                        }
                        CrossValGroup3 = {
                                        loss_function : BCE,
                                        use_positive_weights : true
                                        cross_val_folders_ids : [..]
                                        }
                        }

    """

    evaluated_params, experiment_config_dict = get_dicts_and_varying_params(group_path)

    param_keys = evaluated_params.keys()
    param_list = [evaluated_params[key] for key in param_keys]
    combinations = get_params_combinations(param_list)

    cross_folder_dict = dict()
    for i, combination in enumerate(combinations):
        crossval_group_name = "CrossValGroup: " + str(i)

        cross_folder_dict[crossval_group_name] = dict()

        for k, key in enumerate(param_keys):
            cross_folder_dict[crossval_group_name][key] = combination[k]

        cross_folder_dict[crossval_group_name]["cross_val_folders_ids"] = \
            get_crossval_folders_ids(experiment_config_dict, cross_folder_dict[crossval_group_name])

    return cross_folder_dict


def get_cross_fold_path(dict_list):

    key_values = [current_dict["data_root"] for current_dict in dict_list]
    params_values = list(set(key_values))
    cross_fold_paths = params_values

    return cross_fold_paths
