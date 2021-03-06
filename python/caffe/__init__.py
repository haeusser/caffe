from .pycaffe import Net, SGDSolver, NesterovSolver, Solver, AdamSolver
from ._caffe import set_mode_cpu, set_mode_gpu, set_device, Layer, get_solver, layer_type_list, set_solver_count, get_solver_from_string, P2PSync, set_logging_disabled, setup_teeing 
from .proto.caffe_pb2 import TRAIN, TEST
from .classifier import Classifier
from .detector import Detector
from . import io
from .net_spec import layers, params, NetSpec, to_proto
