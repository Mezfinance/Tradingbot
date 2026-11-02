import sys
from .cli import main

try:
    sys.exit(main())
except KeyboardInterrupt:
    sys.exit(130)
