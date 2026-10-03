#!/usr/bin/env python3
"""Run alignment tests without importing IsaacLab/Gymnasium."""
from pathlib import Path
import sys, types, unittest
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"src"/"gd_lab"
sys.path.insert(0,str(ROOT/"src"))
def ns(name,path):
    mod=types.ModuleType(name); mod.__path__=[str(path)]; mod.__package__=name; sys.modules[name]=mod
ns("gd_lab",SRC); ns("gd_lab.core",SRC/"core"); ns("gd_lab.students",SRC/"students")
ns("gd_lab.students.rvld",SRC/"students"/"rvld"); ns("gd_lab.students.gavd",SRC/"students"/"gavd")
suite=unittest.defaultTestLoader.discover(str(ROOT/"tests"/"students"),pattern="test_alignment_unit.py")
result=unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
