[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13225517.svg)](https://doi.org/10.5281/zenodo.13225517)

# autocode

Auto Code Improvement by Technical Metrics Optimization.

## Description

Autocode selects the best values for optimized metrics. The value types could be bool, int, float, and choice (including
but not limited to code). This project utilizes a Large Language Model and Mixed-Variable Many-Objective Optimization.
Based on our research/literature review, this project hypothetically can contribute to the economic performance of
companies.

## Features

- Value-level Mixed-variable Many-objective Many-software Optimization.
- Value types include bool, int, float, and choice (code).
- Code scoring and variation generators using LLM.
- Software cross-language support.
- Easy software deployment using docker-compose.
- Scalable to infinite cores to speed up processing in parallel.

## How to Use

1. Install the requirements

```bash
pip install -r ./example/requirements.txt
```

2. Prepare software to be processed as in the `./example/client` folder.
3. Prepare deployment as in the `./example/client-compose.yml` file.
3. Prepare controller as in the `./example/controller.ipynb` file.
4. Run the process in controller.
5. Open dashboard in `http://localhost:{dashboard_port}/` to see the process in real-time.
6. Wait until the process is finished.
7. Analyze and decide the best values.

## Demo

- [Controller](./example/controller.ipynb)
- [Client](./example/client)
- Dashboard
  ![demo-1.png](demo-1.png)

## Compatibility

- Python 3.10
- Linux