"""ONNX-runtime backend for badgyal inference (no PyTorch required).

``OnnxNet`` loads an ``.onnx`` file exported from a badgyal PyTorch model
(see ``model.Net.export_onnx``) and runs inference via ``onnxruntime``.  The
plane encoding and policy decoding use numpy variants of ``board2planes`` /
``policy2moves`` so this module has no PyTorch dependency.

Example:
    >>> import badgyal
    >>> net = badgyal.OnnxNet('badgyal-9.onnx')
    >>> policy, value = net.eval(chess.Board())
"""

import os

import numpy as np
import pylru

from badgyal.abstractnet import CACHE, MAX_BATCH, MIN_POLICY, AbstractNet
from badgyal.board2planes import bulk_board2planes_np, policy2moves_np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - optional dependency
    ort = None


class OnnxNet(AbstractNet):
    """Inference backend backed by ONNX Runtime.

    Unlike the PyTorch ``AbstractNet`` subclasses, this class does not call
    ``load_net`` or use ``torch`` at all — the weights live inside the ONNX
    file.  It overrides ``process_boards`` and ``value_to_scalar`` to work
    with numpy arrays.
    """

    def __init__(self, onnx_path: str, softmax_temp: float = 1.61):
        # Skip AbstractNet.__init__ (it loads a PyTorch model).  We only need
        # the cache/prefetch bookkeeping from the base class.
        if ort is None:
            raise ImportError(
                "onnxruntime is required for OnnxNet. "
                "Install it with: pip install onnxruntime"
            )
        self.softmax_temp = softmax_temp
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        self.session = ort.InferenceSession(onnx_path, sess_options=sess_options)
        self.input_name = self.session.get_inputs()[0].name
        # self.net is not used by OnnxNet; set to None to avoid confusion.
        self.net = None
        self.cuda = False
        self.torchScript = False
        self.cache = pylru.lrucache(CACHE)
        self.prefetch = {}

    def process_boards(self, boards):
        """Run the ONNX model on a list of boards.

        Args:
            boards: Iterable of ``chess.Board``.

        Returns:
            A tuple ``(policies, values)`` where ``policies`` is an ndarray
            of shape (N, 1858) and ``values`` is an ndarray of shape (N, 1).
        """
        input_planes = bulk_board2planes_np(boards).astype(np.float32)
        outputs = self.session.run(None, {self.input_name: input_planes})
        # ONNX export order: policy, value (see model.Net.export_onnx).
        policies, values = outputs[0], outputs[1]
        return policies, values

    def value_to_scalar(self, value):
        """Convert a value head output to a scalar.

        For the classical value head the ONNX output is shape (1,) or (1, 1).
        """
        return float(np.asarray(value).reshape(-1)[0])

    def eval(self, board, softmax_temp=1.61):
        """Evaluate a single position.

        Overrides ``AbstractNet.eval`` to avoid the ``torch.no_grad`` /
        ``torch.jit.optimized_execution`` context managers used there.
        """
        epd = board.epd()
        if epd in self.cache:
            policy, value = self.cache[epd]
            return policy, value

        boards = [board.copy()]
        policies, values = self.process_boards(boards)
        for i, b in enumerate(boards):
            policy = policy2moves_np(b, policies[i], softmax_temp=softmax_temp)
            value = self.value_to_scalar(values[i])
            self.cache[b.epd()] = [policy, value]

        policy, value = self.cache[epd]

        # Prefetch likely child positions.
        tocache = []
        for m, val in policy.items():
            if val >= MIN_POLICY:
                bd = board.copy()
                bd.push_uci(m)
                tocache.append(bd)
        if not tocache:
            m = max(policy, key=lambda k: policy[k])
            bd = board.copy()
            bd.push_uci(m)
            tocache.append(bd)
        self.cache_boards(tocache, softmax_temp=softmax_temp)

        return policy, value

    def cache_boards(self, boards, softmax_temp=1.61):
        """Prefetch evaluations for a batch of boards into the cache."""
        for b in boards:
            epd = b.epd()
            if epd not in self.cache:
                self.prefetch[epd] = b

        if len(self.prefetch) > MAX_BATCH:
            to_run = list(self.prefetch.values())
            policies, values = self.process_boards(to_run)
            for i, b in enumerate(to_run):
                policy = policy2moves_np(b, policies[i], softmax_temp=softmax_temp)
                value = self.value_to_scalar(values[i])
                self.cache[b.epd()] = [policy, value]
            self.prefetch = {}

    def bulk_eval(self, boards, softmax_temp=1.61):
        """Evaluate a batch of boards.

        Returns:
            A tuple ``(policies_list, values_list)``.
        """
        retval_p = []
        retval_v = []

        policies, values = self.process_boards(boards)

        for i, b in enumerate(boards):
            policy = policy2moves_np(b, policies[i], softmax_temp=softmax_temp)
            value = self.value_to_scalar(values[i])
            retval_p.append(policy)
            retval_v.append(value)
            self.cache[b.epd()] = [policy, value]

        return retval_p, retval_v
