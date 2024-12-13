
import hashlib

def compute_md5(file_path):
    """
    Compute the MD5 hash of a file.

    :param file_path: Path to the file.
    :return: MD5 hash of the file as a string.
    """
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as file:
            # Read the file in chunks to avoid memory issues with large files
            for chunk in iter(lambda: file.read(4096), b""):
                hash_md5.update(chunk)
    except FileNotFoundError:
        return "File not found."
    except Exception as e:
        return f"An error occurred: {e}"

    # Return the hexadecimal digest of the hash
    return hash_md5.hexdigest()
