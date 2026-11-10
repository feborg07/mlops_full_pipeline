## Installation and Setup

This project uses **Git** for source code versioning and **DVC (Data Version Control)** for dataset versioning. The datasets are stored in a public-read Amazon S3 bucket.

The project can be cloned and its datasets downloaded **without an AWS account or AWS credentials**.

### 1. Prerequisites

Ensure the following tools are installed:

- Python 3.11
- Git
- pip
- DVC with Amazon S3 support

### 2. Clone the Repository

Open a terminal and run:

```bash
git clone https://github.com/USERNAME/mlops_full_pipeline.git
cd mlops_full_pipeline
```

Replace `USERNAME` with the repository owner's GitHub username.

### 3. Create a Virtual Environment

Create an isolated Python environment:

```bash
python -m venv .venv
```

Activate the environment.

**Windows (Git Bash):**

```bash
source .venv/Scripts/activate
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

Install the project's Python dependencies:

```bash
pip install -r requirements.txt
```

Ensure DVC's S3 integration is available:

```bash
pip install "dvc[s3]"
```

### 5. Download the Dataset

The raw dataset is tracked by DVC and stored remotely in Amazon S3.

Download it using:

```bash
dvc pull
```

DVC automatically retrieves the correct dataset