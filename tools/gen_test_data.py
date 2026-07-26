#!/usr/bin/env python3
"""Generate test data for badgyal neural network tests.

This script takes a file of FENs and produces Python test data
in the format expected by tests.py by running lc0 with specific
flags to get policy and value outputs.

Usage:
    python generate_test_data.py <fens_file> [output_file]

    fens_file: Path to a file containing one FEN per line
    output_file: Optional output file (defaults to stdout)
"""

import subprocess
import sys
from pathlib import Path

# Path to your lc0 weights file
LC0_WEIGHTS = Path.home() / "src" / "badgyal" / "badgyal" / "badgyal-9.pb.gz"

# lc0 command flags
LC0_CMD = [
    "lc0",
    "-w",
    str(LC0_WEIGHTS),
    "--verbose-move-stats",
    "--threads=1",
    "--policy-softmax-temp=1.0",
]


def run_lc0(fen: str) -> str:
    """Run lc0 on a given FEN and return the output.

    lc0's search runs asynchronously; if stdin closes (via ``quit``) before
    the search completes, the process exits and the ``info depth`` /
    ``info string`` / ``bestmove`` lines never reach stdout.  We therefore
    drive lc0 interactively with ``Popen`` and wait for ``bestmove`` before
    sending ``quit``.

    Args:
        fen: Chess position in FEN notation.

    Returns:
        Raw output from lc0 stdout.

    Raises:
        RuntimeError: If lc0 fails to execute, times out, or produces no
            ``bestmove`` line.
    """
    # lc0 needs the UCI handshake before it will search.  `go depth 1`
    # produces a single-ply search with verbose move stats; `go nodes 1`
    # emits nothing on some builds.  The weights file is already set via
    # the `-w` flag, so no `setoption` is needed.
    proc = subprocess.Popen(
        LC0_CMD,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        # UCI handshake.
        proc.stdin.write("uci\n")
        proc.stdin.flush()
        # Wait for `uciok` before continuing.
        _read_until(proc.stdout, "uciok", timeout=20)
        proc.stdin.write("isready\n")
        proc.stdin.flush()
        _read_until(proc.stdout, "readyok", timeout=20)

        # Search.
        proc.stdin.write(f"position fen {fen}\ngo depth 1\n")
        proc.stdin.flush()
        search_output = _read_until(proc.stdout, "bestmove", timeout=25)
    finally:
        try:
            proc.stdin.write("quit\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    if "bestmove" not in search_output:
        raise RuntimeError(
            f"lc0 produced no 'bestmove' line. output was: {search_output!r}"
        )

    return search_output


def _read_until(stream, sentinel: str, timeout: float) -> str:
    """Read lines from *stream* until one contains *sentinel*.

    Args:
        stream: A readable text stream (e.g. ``proc.stdout``).
        sentinel: Substring to wait for.
        timeout: Maximum seconds to wait for the sentinel.

    Returns:
        All lines read (joined with newlines), including the matching line.

    Raises:
        RuntimeError: If *sentinel* is not found within *timeout*.
    """
    import time

    deadline = time.monotonic() + timeout
    buf = []
    while time.monotonic() < deadline:
        line = stream.readline()
        if not line:
            break
        buf.append(line.rstrip("\n"))
        if sentinel in line:
            return "\n".join(buf)
    raise RuntimeError(
        f"timed out waiting for '{sentinel}' after {timeout}s; got {len(buf)} lines"
    )


def parse_lc0_output(output: str) -> dict:
    """Parse lc0 output to extract the relevant lines.

    Args:
        output: Raw output from lc0.

    Returns:
        Dictionary with 'header' and 'info_strings' keys.

    Raises:
        RuntimeError: If no `info depth` line is found in the output.
    """
    lines = output.strip().split("\n")

    # Find the header line (starts with "info depth")
    header = None
    info_strings = []

    for line in lines:
        if line.startswith("info depth"):
            header = line
        elif line.startswith("info string") and not line.startswith("info string node"):
            # Skip the "node" line which contains aggregate stats
            info_strings.append(line)
        elif line.startswith("bestmove"):
            break

    if header is None:
        raise RuntimeError(
            "no 'info depth' line found in lc0 output; "
            f"got {len(lines)} lines, first 3: " + " | ".join(lines[:3])
        )

    return {
        "header": header,
        "info_strings": info_strings,
    }


def format_test_entry(fen: str, parsed_output: dict) -> str:
    """Format a single test entry as a Python dictionary entry.

    Args:
        fen: The FEN string.
        parsed_output: Parsed lc0 output dictionary.

    Returns:
        Formatted string for the TESTS dictionary.
    """
    header = parsed_output["header"]
    info_strings = parsed_output["info_strings"]

    # Build the multi-line string value
    value_lines = [header]
    value_lines.extend(info_strings)
    value_str = "\n".join(value_lines)

    # Escape any backslashes and quotes for Python string
    value_str = value_str.replace("\\", "\\\\")
    value_str = value_str.replace('"', '\\"')

    # Format as a Python dictionary entry
    entry = f'    "{fen}" : """\n{value_str}\n""",'

    return entry


def generate_tests(fens: list[str], label: str = "TESTS") -> str:
    """Generate the complete TESTS dictionary string.

    Args:
        fens: List of FEN strings.
        label: Name of the dictionary (e.g., "TESTS" or "ENDGAME_TESTS").

    Returns:
        Complete Python code for the dictionary.
    """
    entries = []
    ok = 0
    failed = 0

    for i, fen in enumerate(fens, 1):
        fen = fen.strip()
        if not fen:
            continue

        print(f"[{i}/{len(fens)}] {fen[:60]}", file=sys.stderr)

        try:
            output = run_lc0(fen)
            parsed = parse_lc0_output(output)
            entry = format_test_entry(fen, parsed)
            entries.append(entry)
            ok += 1
            print(f"  ok ({len(parsed['info_strings'])} moves)", file=sys.stderr)
        except (RuntimeError, subprocess.SubprocessError) as e:
            failed += 1
            print(f"  FAILED: {e}", file=sys.stderr)
            continue

    print(
        f"\nDone: {ok} ok, {failed} failed, {len(fens)} total",
        file=sys.stderr,
    )

    # Build the final output
    result = f"{label} = {{\n"
    result += "\n\n".join(entries)
    result += "\n}"

    return result


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(
            "Usage: python generate_test_data.py <fens_file> [output_file]",
            file=sys.stderr,
        )
        print(
            "  fens_file: Path to a file containing one FEN per line", file=sys.stderr
        )
        print(
            "  output_file: Optional output file (defaults to stdout)", file=sys.stderr
        )
        sys.exit(1)

    fens_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    if not fens_file.exists():
        print(f"Error: File '{fens_file}' not found.", file=sys.stderr)
        sys.exit(1)

    # Read FENs from file
    fens = fens_file.read_text().strip().split("\n")
    fens = [fen.strip() for fen in fens if fen.strip()]

    print(f"Found {len(fens)} FENs to process.", file=sys.stderr)

    # Generate test data
    test_data = generate_tests(fens)

    # Output result
    if output_file:
        output_file.write_text(test_data)
        print(f"Test data written to {output_file}", file=sys.stderr)
    else:
        print(test_data)


if __name__ == "__main__":
    main()
