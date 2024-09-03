[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13225516.svg)](https://doi.org/10.5281/zenodo.13225516)

# autocode

Auto Code Improvement by Metrics Optimization.

## Description

Autocode selects the best values for optimized metrics. The variable value types could be bool, int, float, and choice (including
but not limited to code). This project utilizes a Large Language Model and Mixed-Variable Many-Objective Optimization.
Based on our research/literature review, this project hypothetically can contribute to the economic performance of
companies.  

## Features

- Many-software Value-level Mixed-variable Many-objective Optimization.
- Variable types are bool, int, float, and choice (including but not limited to code).
- Error (MAE/MAE_max) for MCDM (single solution suggestion) is up to 0.0000175.
- Code scoring using LLM.
- Software cross-language support.
- Easy software deployment using docker-compose.
- Scalable to infinite cores to speed up processing in parallel.

## How to Use

1. Install the requirements
- pypi (old)
  ```bash
  pip install -U autocode-py
  ```
- github (new)
  ```bash
  pip install -U git+https://github.com/muazhari/autocode.git@main
  ```

2. Prepare software to be processed as in the [`./example/client`](https://github.com/muazhari/autocode/tree/main/example/client) folder.
3. Prepare deployment as in the [`./example/client/docker-compose.yml`](https://github.com/muazhari/autocode/blob/main/example/client/docker-compose.yml) file.
4. Prepare the controller as in the [`./example/controller.ipynb`](https://github.com/muazhari/autocode/blob/main/example/controller.ipynb) file.
5. Instantiate `optimization` then execute `optimization.deploy()` in the controller.
6. Open the dashboard in `http://localhost:{dashboard_port}/` to see the process in real time.
7. Wait until all clients are ready (need to wait a long time because the libraries need to be re-downloaded for each client).
8. Execute `optimization.run()` in the controller.
9. Wait until the run is finished.
10. Analyze and decide the best values.
11. Execute `optimization.reset(keys=["clients"])` then `optimization.deploy()` to apply different client states.
12. Try to execute `optimization.reset()` to totally reset the tool if needed (i.e. data inconsistency).

## Demo

- [Controller](https://github.com/muazhari/autocode/blob/main/example/controller.ipynb)
- [Client](https://github.com/muazhari/autocode/tree/main/example/client)
- Dashboard
  ![demo-1.png](https://github.com/muazhari/autocode/blob/main/demo-1.png?raw=true)

## Compatibility

- Python 3.10, 3.11, 3.12
- Linux
- Docker
- [autocode-go](https://github.com/muazhari/autocode-go)
