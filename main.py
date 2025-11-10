from src.core.ultra_cortex_simulation import UltraCortexSimulation

def main():
    # Main entry point. Creates and runs a simulated EEG session.
    sim = UltraCortexSimulation()
    sim.start_simulation()

if __name__ == "__main__":
    main()
