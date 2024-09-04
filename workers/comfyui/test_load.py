from lib.test_utils import test_load_cmd, test_args
from .data_types import DefaultComfyWorkflowData, Model

WORKER_ENDPOINT = "/prompt"


if __name__ == "__main__":
    test_args.add_argument(
        "-m",
        dest="comfy_model",
        choices=list(map(lambda x: x.value, Model)),
        required=True,
        help="Image generation model name",
    )
    test_load_cmd(DefaultComfyWorkflowData, WORKER_ENDPOINT, arg_parser=test_args)
