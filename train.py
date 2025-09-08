from __future__ import absolute_import, division, print_function

from options import LiteMonoOptions
from trainer1 import Trainer

options = LiteMonoOptions()
opts = options.parse()


if __name__ == "__main__":
    trainer1 = Trainer(opts)
    trainer1.train()
