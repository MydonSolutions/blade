import math
import numpy as np
import blade as bl

import pandas as pd
import astropy.constants as const
from astropy.coordinates import ITRS, SkyCoord
from astropy.time import Time
import astropy.units as u


@bl.runner
class Pipeline:
    def __init__(self, in_shape, out_shape, config_b, config_gather, config_h):
        self.input.dut = bl.tensor(1, dtype=bl.f64, device=bl.cpu)
        self.input.date = bl.tensor(1, dtype=bl.f64, device=bl.cpu)
        self.input.buffer = bl.array_tensor(in_shape, dtype=bl.cf32)
        self.output.buffer = bl.array_tensor(out_shape, dtype=bl.f32)

        input = (self.input.dut, self.input.date, self.input.buffer)
        self.module.mode_b = bl.module(bl.modeb, config_b, input, telescope=bl.ata)
        self.module.gather = bl.module(bl.gather, config_gather, self.module.mode_b.get_output(), ot=bl.cf32)
        self.module.mode_h = bl.module(bl.modeh, config_h, self.module.gather.get_output(), ot=bl.f32)

    def transfer_in(self, dut, date, buffer):
        self.copy(self.input.dut, dut)
        self.copy(self.input.date, date)
        self.copy(self.input.buffer, buffer)

    def transfer_out(self, buffer):
        self.copy(self.output.buffer, self.module.mode_h.get_output())
        self.copy(buffer, self.output.buffer)


if __name__ == "__main__":
    import os
    script_dir = os.path.dirname(__file__)
    config_bfr5 = {
        "filepath": os.path.join(script_dir, "..", "tests", "blade_input.0000.bfr5")
    }

    config_guppi = {
        "filepath": os.path.join(script_dir, "..", "tests", "blade_input.0000.bfr5"),
        "stepNumberOfTimeSamples": 32,
        "stepNumberOfFrequencyChannels": 1,
        "requiredMultipleOfTimeSamplesSteps": 16,
        "numberOfTimeSampleStepsBeforeFrequencyChannelStep": 0, # read all time before incrementing channel
        "numberOfFilesLimit": 0,
    }
