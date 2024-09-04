from lib.test_utils import test_load_cmd, test_args
from .data_types import InputData

WORKER_ENDPOINT = "/generate"

if __name__ == "__main__":
    test_load_cmd(InputData, WORKER_ENDPOINT, arg_parser=test_args)
