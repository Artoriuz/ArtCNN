import os
import argparse
import numpy as np

def get_file_sizes(extension=".png"):
    """Get the sizes of all files with a specific extension in the current directory."""
    file_sizes = []
    for file in os.listdir("."):
        if file.endswith(extension) and os.path.isfile(file):
            file_sizes.append((file, os.path.getsize(file)))
    return file_sizes

def delete_files_below_percentile(file_sizes, percentile):
    """Delete files below the given percentile size."""
    # Extract sizes and compute the cutoff
    sizes = np.array([size for _, size in file_sizes])
    cutoff = np.percentile(sizes, 100 - percentile)

    # Delete files below the cutoff
    for file, size in file_sizes:
        if size < cutoff:
            print(f"Deleting: {file} (Size: {size} bytes)")
            os.remove(file)

def main(percentile, extension):
    file_sizes = get_file_sizes(extension)
    if not file_sizes:
        print("No files found with the given extension.")
        return

    print(f"Found {len(file_sizes)} files. Processing...")
    delete_files_below_percentile(file_sizes, percentile)
    print("Cleanup complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete files below a given filesize percentile.")
    parser.add_argument("percentile", type=float, help="Percentile of largest files to keep (e.g., 50).")
    parser.add_argument("--extension", default=".png", help="File extension to filter by (default: .png).")

    args = parser.parse_args()
    main(args.percentile, args.extension)
