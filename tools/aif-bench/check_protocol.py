#!/usr/bin/env python3

import argparse
import difflib
import hashlib
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    llama_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Verify that llama.cpp and NVMeVirt use the same AIF wire protocol header."
    )
    parser.add_argument(
        "--nvmevirt-header",
        type=Path,
        default=llama_root.parent / "nvmevirt" / "aif.h",
        help="path to NVMeVirt aif.h (default: sibling nvmevirt repository)",
    )
    args = parser.parse_args()

    llama_header = llama_root / "src" / "llama-aif-proto.h"
    nvmevirt_header = args.nvmevirt_header.resolve()

    try:
        llama_data = llama_header.read_bytes()
        nvmevirt_data = nvmevirt_header.read_bytes()
    except OSError as exc:
        parser.error(str(exc))

    print(f"llama.cpp: {llama_header} sha256={digest(llama_data)}")
    print(f"NVMeVirt:  {nvmevirt_header} sha256={digest(nvmevirt_data)}")
    if llama_data == nvmevirt_data:
        print("AIF protocol headers: MATCH")
        return 0

    print("AIF protocol headers: MISMATCH")
    diff = difflib.unified_diff(
        llama_data.decode("utf-8").splitlines(),
        nvmevirt_data.decode("utf-8").splitlines(),
        fromfile=str(llama_header),
        tofile=str(nvmevirt_header),
        lineterm="",
    )
    for line in diff:
        print(line)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
