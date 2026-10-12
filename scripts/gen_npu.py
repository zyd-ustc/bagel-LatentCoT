"""Launch the fixed C checkpoint GEN color pilot on 16 NPUs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen_latent_cot.diagnostics.gen_parallel import main

if __name__ == '__main__':
    main()
