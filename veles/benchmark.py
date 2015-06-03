"""Veles NN workflow benchmark.
"""
from __future__ import division
import gc
import logging
import numpy
import time
from veles import prng
from veles.backends import CUDADevice, OpenCLDevice
from veles.config import root
from veles.dummy import DummyLauncher
from veles.loader import FullBatchLoader, IFullBatchLoader
from veles.mutable import Bool
from veles.plumbing import Repeater
from veles.units import Unit, IUnit
from veles.znicz.standard_workflow import StandardWorkflow
from zope.interface import implementer


base_lr = 0.01
wd = 0.0005
root.alexnet.update({
    "layers": [{"type": "conv_str",
                "->": {"n_kernels": 64, "kx": 11, "ky": 11,
                       "padding": (2, 2, 2, 2), "sliding": (4, 4),
                       "weights_filling": "gaussian", "weights_stddev": 0.01,
                       "bias_filling": "constant", "bias_stddev": 0},
                "<-": {"learning_rate": base_lr,
                       "learning_rate_bias": base_lr * 2,
                       "weights_decay": wd, "weights_decay_bias": 0,
                       "gradient_moment": 0.9, "gradient_moment_bias": 0.9}},
               {"type": "max_pooling",
                "->": {"kx": 3, "ky": 3, "sliding": (2, 2)}},
               # {"type": "norm", "n": 5, "alpha": 0.0001, "beta": 0.75},

               # {"type": "zero_filter",
               #  "grouping": 2},
               {"type": "conv_str",
                "->": {"n_kernels": 192, "kx": 5, "ky": 5,
                       "padding": (2, 2, 2, 2), "sliding": (1, 1),
                       "weights_filling": "gaussian", "weights_stddev": 0.01,
                       "bias_filling": "constant", "bias_stddev": 0.1},
                "<-": {"learning_rate": base_lr,
                       "learning_rate_bias": base_lr * 2,
                       "weights_decay": wd, "weights_decay_bias": 0,
                       "gradient_moment": 0.9, "gradient_moment_bias": 0.9}},
               {"type": "max_pooling", "->": {"kx": 3, "ky": 3,
                "sliding": (2, 2)}},
               # {"type": "norm", "n": 5, "alpha": 0.0001, "beta": 0.75},

               # {"type": "zero_filter", "grouping": 2},
               {"type": "conv_str",
                "->": {"n_kernels": 384, "kx": 3, "ky": 3,
                       "padding": (1, 1, 1, 1), "sliding": (1, 1),
                       "weights_filling": "gaussian", "weights_stddev": 0.01,
                       "bias_filling": "constant", "bias_stddev": 0},
                "<-": {"learning_rate": base_lr,
                       "learning_rate_bias": base_lr * 2,
                       "weights_decay": wd, "weights_decay_bias": 0,
                       "gradient_moment": 0.9, "gradient_moment_bias": 0.9}},

               # {"type": "zero_filter", "grouping": 2},
               {"type": "conv_str",
                "->": {"n_kernels": 256, "kx": 3, "ky": 3,
                       "padding": (1, 1, 1, 1), "sliding": (1, 1),
                       "weights_filling": "gaussian", "weights_stddev": 0.01,
                       "bias_filling": "constant", "bias_stddev": 0.1},
                "<-": {"learning_rate": base_lr,
                       "learning_rate_bias": base_lr * 2,
                       "weights_decay": wd, "weights_decay_bias": 0,
                       "gradient_moment": 0.9, "gradient_moment_bias": 0.9}},

               # {"type": "zero_filter", "grouping": 2},
               {"type": "conv_str",
                "->": {"n_kernels": 256, "kx": 3, "ky": 3,
                       "padding": (1, 1, 1, 1), "sliding": (1, 1),
                       "weights_filling": "gaussian", "weights_stddev": 0.01,
                       "bias_filling": "constant", "bias_stddev": 0.1},
                "<-": {"learning_rate": base_lr,
                       "learning_rate_bias": base_lr * 2,
                       "weights_decay": wd, "weights_decay_bias": 0,
                       "gradient_moment": 0.9, "gradient_moment_bias": 0.9}},
               {"type": "max_pooling",
                "->": {"kx": 3, "ky": 3, "sliding": (2, 2)}},

               {"type": "all2all",
                "->": {"output_sample_shape": 4096,
                       "weights_filling": "gaussian", "weights_stddev": 0.005,
                       "bias_filling": "constant", "bias_stddev": 1},
                "<-": {"learning_rate": base_lr,
                       "learning_rate_bias": base_lr * 2,
                       "weights_decay": wd, "weights_decay_bias": 0,
                       "gradient_moment": 0.9, "gradient_moment_bias": 0.9}},
               # {"type": "dropout", "dropout_ratio": 0.5},

               {"type": "all2all",
                "->": {"output_sample_shape": 4096,
                       "weights_filling": "gaussian", "weights_stddev": 0.005,
                       "bias_filling": "constant", "bias_stddev": 1},
                "<-": {"learning_rate": base_lr,
                       "learning_rate_bias": base_lr * 2,
                       "weights_decay": wd, "weights_decay_bias": 0,
                       "gradient_moment": 0.9, "gradient_moment_bias": 0.9}},
               # {"type": "dropout", "dropout_ratio": 0.5},

               {"type": "softmax",
                "->": {"output_sample_shape": 1000,
                       "weights_filling": "gaussian", "weights_stddev": 0.01,
                       "bias_filling": "constant", "bias_stddev": 0},
                "<-": {"learning_rate": base_lr,
                       "learning_rate_bias": base_lr * 2,
                       "weights_decay": wd, "weights_decay_bias": 0,
                       "gradient_moment": 0.9, "gradient_moment_bias": 0.9}}
               ]})


@implementer(IFullBatchLoader)
class BenchmarkLoader(FullBatchLoader):
    BATCH = 128

    def load_data(self):
        self.class_lengths[0] = 0
        self.class_lengths[1] = 0
        self.class_lengths[2] = max(self.BATCH, 1000)

        self.create_originals((224, 224, 3))

        prng.get().fill(self.original_data.mem)
        self.original_labels[:] = numpy.arange(
            self.original_data.shape[0], dtype=numpy.int32)


@implementer(IUnit)
class Timer(Unit):
    def __init__(self, workflow, **kwargs):
        super(Timer, self).__init__(workflow, **kwargs)
        self.sync_iterations = set(kwargs.get("sync_iterations", []))
        self.n_it = kwargs.get("n_it", 11)  # one for dry-run
        self.it = 0
        self.complete = Bool(False)
        self.times = [time.time()]

    def initialize(self, **kwargs):
        self.times[:] = [time.time()]
        self.it = 0

    def run(self):
        self.it += 1
        if self.it in self.sync_iterations:
            self.info("%s: Syncing device at iteration %d", self.name, self.it)
            self.workflow.device.sync()
            self.info("%s: Done", self.name)
        if self.it >= self.n_it:
            self.info("%s: Completed, syncing device at iteration %d",
                      self.name, self.it)
            self.workflow.device.sync()
            self.times.append(time.time())
            self.info("%s: Done", self.name)
            self.complete <<= True
        else:
            self.times.append(time.time())


class BenchmarkWorkflow(StandardWorkflow):
    def create_workflow(self):
        self.loader = self.real_loader = BenchmarkLoader(
            self, minibatch_size=BenchmarkLoader.BATCH,
            force_numpy=True)  # do not preload all dataset to device
        self.loader.link_from(self.start_point)

        self.t0 = Timer(self, name="Timer 0",
                        sync_iterations=(1,)).link_from(self.loader)
        self.repeater.link_from(self.t0)
        self.link_forwards(("input", "minibatch_data"), self.loader)
        self.forwards[0].unlink_before()
        self.forwards[0].link_from(self.repeater)
        self.t1 = Timer(self, name="Timer 1",
                        sync_iterations=(1,)).link_from(self.forwards[-2])
        self.repeater.link_from(self.t1)
        self.forwards[-1].gate_block = ~self.t1.complete
        self.forwards[0].gate_block = self.t1.complete

        self.link_evaluator(self.forwards[-1])
        self.link_decision(self.evaluator)
        self.decision.gate_skip = Bool(True)

        last_gd = self.link_gds(self.decision)

        self.t2 = Timer(self, name="Timer 2",
                        sync_iterations=(1,)).link_from(self.gds[-1])
        self.repeater2 = Repeater(self).link_from(self.t2)
        self.gds[-2].unlink_before()
        self.gds[-2].link_from(self.repeater2)

        self.t3 = Timer(self, name="Timer 3",
                        sync_iterations=(1,)).link_from(last_gd)
        self.repeater2.link_from(self.t3)

        self.end_point.link_from(self.t3)
        self.end_point.gate_block = ~self.t3.complete
        self.repeater2.gate_block = self.t3.complete

    def initialize(self, device, **kwargs):
        super(BenchmarkWorkflow, self).initialize(device, **kwargs)
        self.forwards[-1].unlink_before()
        self.forwards[-1].link_from(self.repeater)


root.decision.update({"max_epochs": 1})


def main():
    def nothing():
        pass

    def clBLASOff():
        logging.info("\nclBLAS = OFF\n")
        root.common.engine.ocl.clBLAS = False

    def clBLASOn():
        logging.info("\nclBLAS = ON\n")
        root.common.engine.ocl.clBLAS = True

    for backend in ((CUDADevice, nothing),
                    (OpenCLDevice, clBLASOff),
                    (OpenCLDevice, clBLASOn),
                    ):
        device_class = backend[0]
        backend[1]()
        for dtype in ("float",
                      "double",
                      ):
            logging.info("\n%s: benchmark started for dtype %s\n",
                         device_class, dtype)
            root.common.precision_type = dtype
            root.common.precision_level = 0
            try:
                device = device_class()
            except Exception as e:
                logging.error("Could not create %s: %s", device_class, e)
                break
            launcher = DummyLauncher()
            wf = BenchmarkWorkflow(launcher,
                                   loader_name="imagenet_loader",
                                   decision_config=root.decision,
                                   layers=root.alexnet.layers,
                                   loss_function="softmax")
            wf.initialize(device, snapshot=False)
            if device_class is CUDADevice:
                wf.generate_graph("alexnet.svg")
            wf.run()
            logging.info("Forward pass: %.2f msec",
                         1000.0 * (wf.t1.times[-1] - wf.t1.times[1]) /
                         (wf.t1.n_it - 1))
            logging.info("Backward pass: %.2f msec",
                         1000.0 * (wf.t3.times[-1] - wf.t3.times[1]) /
                         (wf.t3.n_it - 1))

            logging.info("\n%s: benchmark ended for dtype %s\n",
                         device_class, dtype)

            # Full garbage collection
            del wf
            del launcher
            del device
            pool = Unit.reset_thread_pool()
            gc.collect()
            pool.shutdown()
            del pool
            gc.collect()

            # For pypy which can garbage colect with delay
            for _ in range(100):
                gc.collect()

    logging.info("End of job")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
