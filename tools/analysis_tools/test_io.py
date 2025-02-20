from enum import Enum
import types
import inspect
import pprint
import textwrap
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Tuple, Union, Optional

from mmengine.config import Config
import torch
import numpy as np

#  NOTE: You must add class names here so that the input
#   lookup is able to find the file for the new class.
#   The format is "New Class Name":"Old Class Name"
# NAME_MAP = {
#     "MamBEV": "BEVFormerV2",
#     "DetectionTransformerDecoder": "DetectionTransformerDecoder",
#     "PerceptionTransformerV3": "PerceptionTransformerV2",
#     "ResNetFusionV3": "ResNetFusion",
#     "BEVFormerEncoderV3": "BEVFormerEncoder",
#     "DetectionTransformerDecoderV3": "DetectionTransformerDecoder",
#     "BEVFormerHead": "BEVFormerHead",
#     "BEVFormerHeadV3": "BEVFormerHead_GroupDETR",
#     "BEVFormerHead_GroupDETR": "BEVFormerHead_GroupDETR",
#     "BEVFormerHead_GroupDETRV3": "BEVFormerHead_GroupDETR",
#     "FPN": "FPN",
#     "FocalLoss": "FocalLoss",
#     "SmoothL1Loss": "SmoothL1Loss",
#     "L1Loss": "SmoothL1Loss",
#     "ResNet": "ResNet",
#     "LearnedPositionalEncoding": "LearnedPositionalEncoding",
#     "NuscenesDD3D": "NuscenesDD3D",
# }
NAME_MAP = {
    # "MamBEV": "BEVFormer",
    # "DetectionTransformerDecoder": "DetectionTransformerDecoder",
    # "PerceptionTransformerV3": "PerceptionTransformer",
    "ResNetFusionV3": "ResNetFusion",
    "BEVFormerEncoderV3": "BEVFormerEncoder",
    # "DetectionTransformerDecoderV3": "DetectionTransformerDecoder",
    "BEVFormerHead": "BEVFormerHead",
    "BEVFormerHeadV3": "BEVFormerHead",
    "BEVFormerHead_GroupDETR": "BEVFormerHead_GroupDETR",
    "BEVFormerHead_GroupDETRV3": "BEVFormerHead_GroupDETR",
    "FPN": "FPN",
    # "FocalLoss": "FocalLoss",
    # "SmoothL1Loss": "SmoothL1Loss",
    # "L1Loss": "SmoothL1Loss",
    "ResNet": "ResNet",
    "LearnedPositionalEncoding": "LearnedPositionalEncoding",
    # "NuscenesDD3D": "NuscenesDD3D",
}

METHOD_MAP = {
    "predict": "forward_test",
    "loss": "forward",
    "forward": "forward",
    "_forward": "forward",
}


class AtomicityLevel(Enum):
    KEYS_ONLY = 0
    SHAPE_TYPE = 1
    VALUE = 2


def print_bordered(message: str, top: str = "~", sides: str = " "):
    message_lines = []
    print(top * 80)
    for line in message.splitlines():
        message_lines.extend(
            textwrap.wrap(
                line,
                tabsize=4,
                initial_indent=" " * 2,
                subsequent_indent=" " * 6,
            )
        )
    lines = (" ",) + tuple(message_lines) + (" ",)

    for m in lines:
        print(f"{sides} {m:<76} {sides}")
    print(top * 80, end="\n\n")


def print_error_counts(counts: Dict[str, int], which: str):
    total_errors = sum(counts.values())
    if total_errors == 0:
        print_bordered(f"Passed all assertions in {which}")
    else:
        message = f"{total_errors} Errors Found in {which}\n"
        for e_type, count in counts.items():
            message += f"{e_type:>20}: {count}\n"
        print_bordered(message, top="=", sides=" ")


def check_log_assertion(condition, message):
    try:
        assert condition
    except AssertionError:
        print_bordered(message)
        return 1
    return 0


def traverse_compare(
    gt: Any,
    data: Any,
    prefix: str,
    compare_atomic: Union[int, AtomicityLevel] = AtomicityLevel.SHAPE_TYPE,
    num_errors: Dict[str, int] = {
        "Different Type Errors": 0,
        "Different Length or Shape Errors": 0,
        "Different Keys Errors": 0,
        "Different Dtype Errors": 0,
        "Different Value Errors": 0,
    },
):
    type_gt, type_data = type(gt), type(data)
    type_error = check_log_assertion(
        type_gt == type_data,
        (
            prefix
            + "\n"
            + f"GT and Data are different types\nGT:={type_gt} != Data:={type_data}"
            # + f"\n{gt}"
            # + f"\n{data}"
        ),
    )
    num_errors["Different Type Errors"] += type_error
    list_like = (List, Tuple)

    dict_like = (dict,)
    data_like = np.array(
        (isinstance(data, list_like), isinstance(data, dict_like)), dtype=bool
    )
    gt_like = np.array(
        (isinstance(gt, list_like), isinstance(gt, dict_like)), dtype=bool
    )
    if type_error and not np.array_equal(data_like, gt_like):
        print("Skipping checks due to prior error")
        pass

    elif data_like[0]:
        num_errors["Different Length or Shape Errors"] += check_log_assertion(
            len(data) == len(gt),
            (
                prefix
                + "\n"
                + f"GT and Data are iterables of different lengths\n\tGT:={len(gt)} != Data:={len(data)}"
            ),
        )
        for i, (g, d) in enumerate(zip(gt, data)):
            traverse_compare(
                gt=g,
                data=d,
                prefix=prefix + f"[{repr(i)}]",
                compare_atomic=compare_atomic,
                num_errors=num_errors,
            )
    elif data_like[1]:
        keys_data = set(data.keys())
        keys_gt = set(gt.keys())
        key_diff_gt = keys_data - keys_gt
        key_diff_data = keys_gt - keys_data
        num_errors["Different Keys Errors"] += check_log_assertion(
            not (key_diff_data or key_diff_gt),
            (
                prefix
                + "\n"
                + f"{key_diff_data} are missing from Data and {key_diff_gt} are missing from GT"
            ),
        )
        for k in keys_data & keys_gt:
            traverse_compare(
                gt=gt[k],
                data=data[k],
                prefix=prefix + f"[{repr(k)}]",
                compare_atomic=compare_atomic,
                num_errors=num_errors,
            )

    elif compare_atomic == AtomicityLevel.SHAPE_TYPE or compare_atomic == 1:
        if isinstance(data, torch.Tensor):
            ddev = data.device
            gdev = gt.device
            if gdev != ddev:
                gt_copy = gt.cpu()
                data_copy = data.cpu()
            else:
                gt_copy = gt
                data_copy = data
            num_errors["Different Length or Shape Errors"] += check_log_assertion(
                data_copy.size() == gt_copy.size(),
                (
                    prefix
                    + "\n"
                    + f"Data and GT tensors are not the same shape\n\tData:={data.size()} != GT:={gt.size()}"
                ),
            )
            num_errors["Different Dtype Errors"] += check_log_assertion(
                data_copy.type() == gt_copy.type(),
                (
                    prefix
                    + "\n"
                    + f"Data and GT tensors are not the same dtype\n\tData:={data.type()} != GT:={gt.type()}"
                ),
            )

        elif isinstance(data, np.ndarray):
            num_errors["Different Length or Shape Errors"] += check_log_assertion(
                len(data) == len(gt),
                (
                    prefix
                    + "\n"
                    + f"Data and GT arrays are not the same size\n\tData:={len(data)} != GT:={len(gt)}"
                ),
            )
            num_errors["Different Length or Shape Errors"] += check_log_assertion(
                np.array_equal(data.shape, gt.shape),
                (
                    prefix
                    + "\n"
                    + f"Data and GT arrays are not the same shape\n\tData:={data.shape} != GT:={gt.shape}"
                ),
            )

            if len(data):
                num_errors["Different Dtype Errors"] += check_log_assertion(
                    data[0].dtype == gt[0].dtype,
                    (
                        prefix
                        + "\n"
                        + f"Data and GT arrays are not the same dtype\n\tData:={data.dtype} != GT:={gt.dtype}"
                    ),
                )

    elif compare_atomic == AtomicityLevel.VALUE or compare_atomic == 2:
        if isinstance(data, (int, float, str)):
            num_errors["Different Value Errors"] += check_log_assertion(
                gt == data,
                (prefix + "GT and Data are not equal" + f"\n\t{gt} == {data}" + "\n"),
            )

        elif isinstance(data, torch.Tensor):
            ddev = data.device
            gdev = gt.device
            if gdev != ddev or gdev != "cpu":
                gt_copy = gt.cpu()
                data_copy = data.cpu()
            else:
                gt_copy = gt
                data_copy = data

            if data_copy.type() == torch.FloatTensor:
                num_errors["Different Value Errors"] += check_log_assertion(
                    torch.allclose(gt_copy, data_copy),
                    prefix + "\nDifferent Value Error",
                )
            else:
                num_errors["Different Value Errors"] += check_log_assertion(
                    torch.equal(gt_copy, data_copy), prefix + "\nDifferent Value Error"
                )
        elif isinstance(data, np.ndarray):
            if len(data) and len(gt):
                if isinstance(data[0], np.floating):
                    num_errors["Different Value Errors"] += check_log_assertion(
                        np.isclose(gt, data).all(), prefix + "\nDifferent Value Error"
                    )
                else:
                    num_errors["Different Value Errors"] += check_log_assertion(
                        np.array_equal(gt, data), prefix + "\nDifferent Value Error"
                    )
        else:
            print(type(data), type(gt))
            raise NotImplementedError(prefix + "Unhandled type")
    return num_errors


def get_argument_names(method):
    signature = inspect.signature(method)

    pos_args = []
    kw_args = []

    for name, param in signature.parameters.items():
        if param.default == inspect.Parameter.empty and param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            pos_args.append(name)
        elif (
            param.default != inspect.Parameter.empty
            or param.kind == inspect.Parameter.KEYWORD_ONLY
        ):
            kw_args.append(name)

    return pos_args, kw_args


def compare_to_original(
    method,
    save_dir: Union[str, Path] = "io",
    check_inputs: bool = True,
    check_outputs: bool = True,
    compare_atomic: Union[int, AtomicityLevel] = AtomicityLevel.SHAPE_TYPE,
    run_type: Optional[Union[Literal["train"], Literal["test"]]] = "train",
    input_translator: Optional[Callable[..., Dict]] = None,
):
    save_dir = Path(save_dir)
    assert save_dir.exists(), "Path to input data not found"
    pos_arg_names, kw_arg_names = get_argument_names(method)
    method_name = method.__name__
    class_name = method.__class__.__name__
    try:
        old_class_name = NAME_MAP[class_name]
    except KeyError:
        print(
            f"WARNING: Failed to wrap {class_name} as class name not"
            " found in name map. If this is unintended add the class name"
            " to the name map in test_io.py"
        )
        return method
    try:
        old_method_name = METHOD_MAP[method_name]
    except KeyError:
        print(
            f"WARNING: Failed to wrap {class_name}.{method_name} as {method_name} name not"
            " found in method map. If this is unintended add the method name"
            " to the method map in test_io.py"
        )
        return method

    def wrapper(self, *args, **kwargs):
        input_data = [args, kwargs]
        input_data = {k: v for k, v in zip(pos_arg_names, args)}
        input_data.update(kwargs)
        pprint.pp(kwargs, depth=3)
        if check_inputs:
            input_file = save_dir / f"{old_class_name}_{old_method_name}_input.pkl"
            if not input_file.exists():
                input_file = (
                    save_dir
                    / f"{old_class_name}_{old_method_name}__{run_type}_input.pkl"
                )
            # load gt inputs
            with open(input_file, "rb") as f:
                input_data_old = pickle.load(f)
            input_data_gt = {k: v for k, v in zip(pos_arg_names, input_data_old[0])}
            input_data_gt.update(input_data_old[1])
            # translate arguments
            if input_translator is not None:
                input_data_gt = input_translator(class_name, **input_data_gt)
            # Traverse input and compare
            errors = traverse_compare(
                input_data_gt,
                input_data,
                prefix=f"{class_name}.{method_name}",
                compare_atomic=compare_atomic,
                num_errors={
                    "Different Type Errors": 0,
                    "Different Length or Shape Errors": 0,
                    "Different Keys Errors": 0,
                    "Different Dtype Errors": 0,
                    "Different Value Errors": 0,
                },
            )
            print_error_counts(errors, f"input of {class_name}.{method_name}")
        # Get module outputs
        try:
            output_data = method(*args, **kwargs)
        except TypeError as e:
            print("Arg names")
            pprint.pp(pos_arg_names + kw_arg_names, indent=4)
            print("Inputs")
            pprint.pp({"args": args, "kwargs": kwargs}, indent=4, depth=2)
            print(f"Args Error in {class_name}.{method_name}... Exiting")
            raise e

        if check_outputs:
            output_file = save_dir / f"{old_class_name}_{old_method_name}_output.pkl"
            if not output_file.exists():
                output_file = (
                    save_dir
                    / f"{old_class_name}_{old_method_name}__{run_type}_output.pkl"
                )

            # load gt output
            with open(output_file, "rb") as f:
                output_data_gt = pickle.load(f)
            # traverse output and compare
            errors = traverse_compare(
                output_data_gt,
                output_data,
                prefix=f"{class_name}.{method_name}  ",
                compare_atomic=compare_atomic,
                num_errors={
                    "Different Type Errors": 0,
                    "Different Length or Shape Errors": 0,
                    "Different Keys Errors": 0,
                    "Different Dtype Errors": 0,
                    "Different Value Errors": 0,
                },
            )
            print_error_counts(errors, f"output of {class_name}.{method_name}")

        return output_data

    return wrapper


def wrap_forward_methods(
    module,
    config: Union[Dict, Config],
    save_dir: Union[str, Path],
    check_inputs: bool = True,
    check_outputs: bool = False,
    compare_atomic: Union[int, AtomicityLevel] = AtomicityLevel.SHAPE_TYPE,
    input_translator: Optional[Callable[..., Dict]] = None,
    run_type: Optional[Union[Literal["train"], Literal["test"]]] = "train",
):
    forward_method_names = (
        "forward",
        "decode",
        "encode",
        "loss",
        "predict",
        "forward_test",
        "forward_train",
    )

    def replace_with_wrapped(
        module,
        config: Union[Dict[str, Any], Config],
    ):
        for att in forward_method_names:
            if hasattr(module, att):
                setattr(
                    module,
                    att,
                    types.MethodType(
                        compare_to_original(
                            getattr(module, att),
                            save_dir=save_dir,
                            check_inputs=check_inputs,
                            check_outputs=check_outputs,
                            compare_atomic=compare_atomic,
                            input_translator=input_translator,
                            run_type=run_type,
                        ),
                        module,
                    ),
                )
        submod_names = {
            m["type"].split(".")[-1]: m
            for m in config.values()
            if isinstance(m, dict) and "type" in m
        }
        # Recursively wrap the forward methods of all submodules
        for submodule in module.children():
            sname = submodule.__class__.__name__

            if sname in submod_names:
                replace_with_wrapped(
                    submodule,
                    submod_names[sname],
                )

    replace_with_wrapped(module, config)
