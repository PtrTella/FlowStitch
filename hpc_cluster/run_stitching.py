import sys
from flowstitch.core.config import FlowStitchConfig
from flowstitch.pipelines.latent_stitching import run_latent_stitching

if __name__ == "__main__":
    config = FlowStitchConfig()
    # Esempio di come sovrascrivere valori
    config.stitching_mode = "full"
    
    db_path = "data/dataset_v1/a_red_cube"
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
        
    run_latent_stitching(config, db_path)
