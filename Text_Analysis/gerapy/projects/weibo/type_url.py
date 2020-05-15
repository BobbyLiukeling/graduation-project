# -*- encoding: utf-8 -*-
# @author : bobby
# @time : 2020/3/5 20:32

from scrapy.cmdline import execute

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
execute(["scrapy", "crawl", "film_type"])