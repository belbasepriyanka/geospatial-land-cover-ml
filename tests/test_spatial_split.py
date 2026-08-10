import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from spatial_split import spatial_block_ids

def test_blocks():
    ids = spatial_block_ids([0,5,25],[0,5,25],20)
    assert ids[0] == ids[1]
    assert ids[2] != ids[0]
