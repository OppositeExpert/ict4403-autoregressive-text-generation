"""Optional recovery of archived dataset; never silently replace a changed source."""
from pathlib import Path
import hashlib
from urllib.request import urlopen
from .data import DOWNLOAD_URL, EXPECTED_SHA256


def main():
    path=Path("data/raw/alice_pg11.txt")
    if path.exists():
        raw=path.read_bytes()
    else:
        with urlopen(DOWNLOAD_URL,timeout=60) as response:
            raw=response.read()
    checksum=hashlib.sha256(raw).hexdigest()
    if checksum != EXPECTED_SHA256:
        raise SystemExit(f"Upstream text differs ({checksum}). Use the version archived with this project.")
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        path.write_bytes(raw)
    print(f"Verified {path}: {checksum}")


if __name__=="__main__":
    main()
