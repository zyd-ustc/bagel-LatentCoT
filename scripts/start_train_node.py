"""Start one torchrun process detached from the SSH session."""

import argparse
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--pid-file")
    args, command = parser.parse_known_args()
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("A command is required after --")
    with open(args.log, "ab") as log:
        process = subprocess.Popen(
            command,
            cwd=os.getcwd(),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    if args.pid_file:
        Path(args.pid_file).write_text(str(process.pid) + "\n")
    print(process.pid, flush=True)


if __name__ == "__main__":
    main()
