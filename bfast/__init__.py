from .models import BFASTMonitor
from warnings import warn

__version__ = '0.8.dev0'

# check that cupy is installed 
try: 
    import cupy 
except ModuleNotFoundError:
    warn("cupy is not available in this environment, GPU fonctionnalities won't be available")
