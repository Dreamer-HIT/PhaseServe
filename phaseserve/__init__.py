import os
import torch

if not os.environ.get("SWIFT_TRANSFORMER_LIB_PATH"):
    BASE_DIR = os.path.dirname(__file__)
    LIB_PATH = os.path.join(BASE_DIR, "../SwiftTransformer/build/lib", "libst_pybinding.so")
else:
    LIB_PATH = os.environ["SWIFT_TRANSFORMER_LIB_PATH"]

if not os.path.exists(LIB_PATH):
    raise RuntimeError(
        f"Could not find the SwiftTransformer library libst_pybinding.so at {LIB_PATH}. "
        "Please build the SwiftTransformer library first or put it at the right place."
    )

torch.ops.load_library(LIB_PATH)

from phaseserve.llm import OfflineLLM
from phaseserve.request import SamplingParams
