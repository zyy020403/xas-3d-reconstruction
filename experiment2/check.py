# 在命令行运行这个，直接测试
import sys
sys.path.insert(0, r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment2\step2")
from spectrum_preprocessor import load_xmu_xanes
import inspect
print(inspect.getsourcefile(load_xmu_xanes))  # 确认用的是哪个文件