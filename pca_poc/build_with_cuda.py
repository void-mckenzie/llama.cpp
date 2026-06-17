#!/usr/bin/env python3
"""
Configure and build llama.cpp with CUDA support.

Idempotent: skips cmake configure if build dir is already configured with
matching options, and skips targets whose binaries are up-to-date.

Usage:
    python build_with_cuda.py
    python build_with_cuda.py --llama_dir F:/Repos/llama.cpp --build_dir F:/Repos/llama.cpp/build-cuda
    python build_with_cuda.py --cuda_arch 89 --targets llama-cli llama-perplexity
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def find_cmake():
    cmake = shutil.which("cmake")
    if not cmake:
        sys.exit("ERROR: cmake not found on PATH. Install CMake and re-run.")
    return cmake


def find_nvgencat_or_nvcc():
    """Quick CUDA toolkit presence check."""
    return shutil.which("nvcc") is not None


def is_already_configured(build_dir: Path, llama_dir: Path, cuda_arch: str) -> bool:
    cache = build_dir / "CMakeCache.txt"
    if not cache.exists():
        return False
    text = cache.read_text(encoding="utf-8", errors="replace")
    checks = [
        f"GGML_CUDA:BOOL=ON" in text,
        f"CMAKE_CUDA_ARCHITECTURES:UNINITIALIZED={cuda_arch}" in text or
        f"CMAKE_CUDA_ARCHITECTURES:STRING={cuda_arch}" in text,
        str(llama_dir) in text,
    ]
    return all(checks)


def target_up_to_date(build_dir: Path, target: str, config: str = "Debug") -> bool:
    """Check if the target's binary exists. (Doesn't check freshness —
    cmake --build will do incremental compilation anyway.)"""
    bin_dir = build_dir / "bin" / config
    exe = bin_dir / f"{target}.exe"
    return exe.exists()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--llama_dir", type=Path, default=Path("F:/Repos/llama.cpp"),
                   help="llama.cpp repo root")
    p.add_argument("--build_dir", type=Path, default=Path("F:/Repos/llama.cpp/build-cuda"),
                   help="Build directory")
    p.add_argument("--cuda_arch", default="89",
                   help="CUDA architecture (89 = RTX 3500 Ada, 86 = RTX 30/40 series, 80 = A100)")
    p.add_argument("--targets", nargs="+", default=["llama-cli", "llama-perplexity"],
                   help="CMake targets to build")
    p.add_argument("--config", default="Debug",
                   help="Build config (Debug/Release). MSVC multi-config puts binaries in bin/<config>/.")
    p.add_argument("--fresh", action="store_true",
                   help="Re-configure from scratch (delete CMakeCache.txt first)")
    p.add_argument("--shared", action="store_true", default=True,
                   help="Build shared libs (default: on, matches the existing setup)")
    args = p.parse_args()

    cmake = find_cmake()
    if not find_nvgencat_or_nvcc():
        print("WARN: nvcc not found on PATH. CUDA toolkit may not be installed.")

    if not args.llama_dir.exists():
        sys.exit(f"ERROR: llama_dir not found: {args.llama_dir}")

    args.build_dir.mkdir(parents=True, exist_ok=True)

    # ===== Phase 1: configure (if needed) =====
    need_configure = (
        args.fresh
        or not is_already_configured(args.build_dir, args.llama_dir, args.cuda_arch)
    )
    if need_configure:
        if args.fresh and (args.build_dir / "CMakeCache.txt").exists():
            print(f"  --fresh: removing existing CMakeCache.txt")
            (args.build_dir / "CMakeCache.txt").unlink()

        print(f"\n=== Phase 1: cmake configure ===")
        cmd = [
            cmake,
            "-S", str(args.llama_dir),
            "-B", str(args.build_dir),
            "-DGGML_CUDA=ON",
            f"-DCMAKE_CUDA_ARCHITECTURES={args.cuda_arch}",
            "-DBUILD_SHARED_LIBS=ON" if args.shared else "-DBUILD_SHARED_LIBS=OFF",
            f"-DCMAKE_BUILD_TYPE={args.config}",
        ]
        print(f"  cmd: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            sys.exit(f"ERROR: cmake configure failed (rc={result.returncode})")
    else:
        print(f"  {args.build_dir} already configured for arch={args.cuda_arch}, skipping")

    # ===== Phase 2: build =====
    print(f"\n=== Phase 2: build targets: {args.targets} ===")
    cmd = [cmake, "--build", str(args.build_dir), "--config", args.config, "-j"]
    for t in args.targets:
        cmd += ["--target", t]
    print(f"  cmd: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"ERROR: build failed (rc={result.returncode})")

    # ===== Verify =====
    print(f"\n=== Phase 3: verify ===")
    bin_dir = args.build_dir / "bin" / args.config
    all_ok = True
    for t in args.targets:
        exe = bin_dir / f"{t}.exe"
        if exe.exists():
            size_mb = exe.stat().st_size / 1024 / 1024
            print(f"  OK   {exe}  ({size_mb:.1f} MB)")
        else:
            print(f"  MISS {exe}")
            all_ok = False

    # CUDA DLLs should be in same dir (copied by build system) or in PATH
    cuda_dlls = list(bin_dir.glob("ggml-cuda*.dll"))
    if cuda_dlls:
        for dll in cuda_dlls:
            print(f"  OK   {dll.name}")
    else:
        print(f"  WARN: no ggml-cuda*.dll found in {bin_dir}")

    if all_ok:
        print(f"\n=== BUILD DONE ===")
        print(f"Binaries at: {bin_dir}")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
