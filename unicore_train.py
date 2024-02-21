#!/data/protein/SKData/miniconda3/envs/MOL/bin/python
# -*- coding: utf-8 -*-
import re
import sys
from unicore_cli.train import cli_main
if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(cli_main())