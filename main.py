from src.core.ultra_cortex_simulation import UltraCortexSimulation

def main():
    # Create the simulation object
    sim = UltraCortexSimulation(config_path="configs/app_config.yaml")

    # Start the simulation
    sim.start_simulation()

    # Print the final input source (after fallback if needed)
    print(f"[Main] Input source now running as: {sim.mode}")
    print(f"[Main] Data source type: {type(sim.data_source).__name__}")

if __name__ == "__main__":
    main()