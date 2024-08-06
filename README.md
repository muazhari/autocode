[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13225517.svg)](https://doi.org/10.5281/zenodo.13225517)

# autocode

Auto Code Improvement by Metrics Optimization.

## Description

Autocode selects the best values for optimized metrics. The value types could be bool, int, float, and choice (including
but not limited to code). This project utilizes a Large Language Model and Mixed-Variable Many-Objective Optimization.
Based on our research/literature review, this project hypothetically can contribute to the economic performance of
companies.

## Features

- Many-software Value-level Mixed-variable Many-objective Optimization.
- Value types include bool, int, float, and choice (code).
- Code scoring and variation generators using LLM.
- Software cross-language support.
- Easy software deployment using docker-compose.
- Scalable to infinite cores to speed up processing in parallel.

## How to Use

1. Install the requirements

```bash
pip install autocode-py
```

2. Prepare software to be processed as in the [`./example/client`](https://github.com/muazhari/autocode/tree/main/example/client) folder.
3. Prepare deployment as in the [`./example/client/docker-compose.yml`](https://github.com/muazhari/autocode/blob/main/example/client/docker-compose.yml) file.
4. Prepare controller as in the [`./example/controller.ipynb`](https://github.com/muazhari/autocode/blob/main/example/controller.ipynb) file.
5. Instantiate `optimization` and execute `optimization.deploy()` in controller.
6. Open dashboard in `http://localhost:{dashboard_port}/` to see the process in real-time.
7. Wait until all client are ready (need to wait for long time because the libraries need to be re-download for each client & refresh the dashboard manually because of bug).
8. Execute `optimization.run()` in controller.
9. Wait until the run is finished.
10. Analyze and decide the best values.

## Demo

- [Controller](https://github.com/muazhari/autocode/blob/main/example/controller.ipynb)
- [Client](https://github.com/muazhari/autocode/tree/main/example/client)
- Dashboard
  ![demo-1.png](https://github.com/muazhari/autocode/blob/main/demo-1.png?raw=true)

## Compatibility

- Python 3.10, 3.11
- Linux
- Docker
- [autocode-go](https://github.com/muazhari/autocode-go)
