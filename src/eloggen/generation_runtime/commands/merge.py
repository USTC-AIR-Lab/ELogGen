"""
Merge per-episode generated HDF5 files without running simulation.
"""
import argparse
import contextlib
import os

import eloggen.generation_runtime.datasets as DatasetUtils


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-episode HDF5 files from a generated dataset tmp folder."
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="Folder containing per-episode .hdf5 files, usually .../demo_src_xxx/tmp",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Final merged HDF5 path, usually .../demo_src_xxx/demo.hdf5",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output if it already exists. Merge is written to a temporary file first.",
    )
    parser.add_argument(
        "--delete-folder",
        action="store_true",
        help="Delete the input folder after a successful merge.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only count valid input files and print horizon stats.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show verbose output from the underlying merge utility.",
    )
    args = parser.parse_args()

    folder = os.path.abspath(os.path.expanduser(args.folder))
    output = os.path.abspath(os.path.expanduser(args.output))

    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Input folder does not exist: {folder}")

    @contextlib.contextmanager
    def maybe_quiet():
        if args.verbose:
            yield
        else:
            with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
                yield

    if args.dry_run:
        with maybe_quiet():
            num_files, horizons = DatasetUtils.merge_all_hdf5(
                folder=folder,
                new_hdf5_path=output,
                dry_run=True,
                return_horizons=True,
            )
        total = int(sum(horizons))
        print(f"[merge] valid files: {num_files}")
        print(f"[merge] total steps: {total}")
        if horizons:
            print(
                "[merge] horizon min/mean/max: "
                f"{min(horizons)} / {sum(horizons) / len(horizons):.1f} / {max(horizons)}"
            )
        return

    if os.path.exists(output) and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output}\n"
            "Pass --overwrite to replace it after a successful temporary merge."
        )

    os.makedirs(os.path.dirname(output), exist_ok=True)
    tmp_output = output + ".merging"
    if os.path.exists(tmp_output):
        raise FileExistsError(f"Temporary merge output already exists: {tmp_output}")

    try:
        with maybe_quiet():
            merged = DatasetUtils.merge_all_hdf5(
                folder=folder,
                new_hdf5_path=tmp_output,
                delete_folder=False,
            )
        os.replace(tmp_output, output)
    finally:
        if os.path.exists(tmp_output):
            os.remove(tmp_output)

    if args.delete_folder:
        import shutil

        shutil.rmtree(folder)

    print(f"[merge] merged {merged} files -> {output}")
    if args.delete_folder:
        print(f"[merge] deleted input folder: {folder}")


if __name__ == "__main__":
    main()
