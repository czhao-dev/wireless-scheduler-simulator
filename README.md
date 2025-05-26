# MAC Layer Scheduling and Resource Allocation Simulator

## Overview

This project simulates MAC layer scheduling and resource allocation at a base station in a cellular network, handling downlink traffic for mobile stations (MS) under different scheduling schemes. The simulation models packet buffering, queuing, transmission, and fairness mechanisms to analyze throughput and delay performance across user classes.

## Features

- Simulates three flow classes with distinct Quality of Service (QoS) requirements.
- Implements and compares three scheduling algorithms:
  - **Priority Oriented (PO)**
  - **Weighted Round Robin (WRR)**
  - **WRR + Proportional Fair Throughput (WRR+PFT)**
- Evaluates throughput, packet delay, and fairness.
- Generates visualizations to support performance comparison.
- Modular Python 3 implementation with customizable parameters.

## Flow Classes

| Class | Priority | QoS Throughput | QoS Delay | Packet Size | Traffic Pattern |
|-------|----------|----------------|-----------|--------------|------------------|
| 1     | High     | 50 Kbps        | ≤ 60 ms   | 480 bits     | 400ms burst / 600ms pause |
| 2     | Medium   | 1 Mbps         | ≤ 360 ms  | 1600 bits    | 1s burst / 4s pause |
| 3     | Low      | 0.4 Mbps       | ~600 ms   | 1200 bits    | Poisson Process |

## Scheduling Algorithms

1. **Priority Oriented (PO)**: Assigns time slots based on class priority, potentially favoring flows with high CQI.
2. **Weighted Round Robin (WRR)**: Allocates slots proportionally to class throughput weights, cyclically without CQI consideration.
3. **WRR + PFT**: Enhances WRR with a proportional fairness scheduler, balancing throughput and delay across MSs.

## Performance Metrics

- Maximum number of supported flows (`Nmax`)
- Average throughput per class
- Average and standard deviation of packet delays
- Fairness in throughput and delay among mobile stations

## Simulation

- Time granularity: 1ms time slots
- Bandwidth: 20 MHz downlink channel
- 8 mobile stations with varying spectral efficiency:
  - 40% @ 0.2 bps/Hz
  - 30% @ 1 bps/Hz
  - 30% @ 2 bps/Hz
- Each flow is mapped to a random MS.

## How to Run

```bash
python simulation.py
```

### Output

- Simulation results are stored in `.pcy` pickle files.
- Use `Unpacker` class to load and visualize simulation results.

## File Structure

- `simulation.py`: Contains all classes and logic for simulation and plotting.
- `PO_simulation_data.pcy`, `WRR_simulation_data.pcy`, `WRR+PFT_simulation_data.pcy`: Pickle files storing simulation results.

## Key Takeaways

- **WRR+PFT** outperforms the other two schemes in terms of fairness and delay control.
- Priority-based methods may lead to unfair resource distribution.
- Proportional fairness enhances service to low-throughput users with long wait times.

## Dependencies

- Python 3
- `matplotlib`, `numpy`, `statistics`, `pickle`

## License

This project is released under the Apache 2.0 License.
