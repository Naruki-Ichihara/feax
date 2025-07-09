from pyfiglet import Figlet

f = Figlet(font='starwars')
print(f.renderText('FEAX'))

from .logger_setup import setup_logger
# LOGGING
logger = setup_logger(__name__)

# Import main modules
from . import apply_bcs
from .apply_bcs import apply_dirichletBC, BCInfo, DirichletBC, create_bc_info, create_fixed_bc

__version__ = "0.0.1"