import logging

from tradingbot.cli import main

try:
    main()
except KeyboardInterrupt:
    # Fly stops machines with SIGINT on deploy; exit quietly instead of printing a traceback.
    logging.getLogger("tradingbot").info("interrupted, shutting down")
    raise SystemExit(130)
