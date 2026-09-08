"""Terminal UI.

Imported lazily by the CLI so that ``quantdesk backtest`` does not need Textual
installed or a TTY available.
"""

from quantdesk.ui.app import QuantDeskApp

__all__ = ["QuantDeskApp"]
