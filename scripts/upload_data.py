import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transpetro_modelos.data.upload_datasets import main

if __name__ == "__main__":
    main()
