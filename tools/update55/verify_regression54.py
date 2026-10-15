"""Run inherited UPDATE54 unit suite; redirect output without changing old artifacts."""
from pathlib import Path
root=Path(__file__).resolve().parents[2]
source=(root/'tools/update54/verify_cpu.py').read_text().replace("OUT=ROOT/'outputs/update54_cpu'", "OUT=ROOT/'outputs/update55_cpu/regression54'")
exec(compile(source,str(root/'tools/update54/verify_cpu.py'),'exec'),{'__name__':'__main__','__file__':str(root/'tools/update54/verify_cpu.py')})
