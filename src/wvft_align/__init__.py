"""Wavefront alignment utility."""


def main() -> None:
    """Start the application with an actionable error when Tk is unavailable."""
    try:
        import tkinter
    except ModuleNotFoundError as exc:
        if exc.name not in {"tkinter", "_tkinter"}:
            raise
        raise SystemExit(
            "wvft-align requires Tkinter, which is not installed.\n"
            "On Debian/Ubuntu, install it with:\n"
            "  sudo apt install python3-tk"
        ) from None

    from wvft_align.main import main as run

    run()
