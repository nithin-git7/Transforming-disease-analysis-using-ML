# MediAI

MediAI is a Streamlit portfolio project maintained by [nithin-git7](https://github.com/nithin-git7) that demonstrates four machine-learning classifiers. The demo accepts chest X-ray and mammogram samples, plus manually entered or CSV-based examples for diabetes and chronic kidney disease. It displays model labels and scores and generates educational PDF reports.

Use synthetic records or publicly available sample images. The models were built for education and have no established clinical validity. Scores are model outputs and should never guide medical decisions.

## Run locally

Install Python 3.10 and Git LFS. In the project directory, run the following commands.

```sh
git lfs install
git lfs pull
python -m venv .venv
```

Activate the environment with `.\.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` in Bash. Then install the dependencies and launch the app.

```sh
python -m pip install -r requirements.txt
python -m streamlit run Home.py
```

Open [the local app](http://localhost:8501). Sign-in is optional. Signed-in users can view and export their own prediction history. The default SQLite database is created on first use and ignored by Git. New history entries store the model label, scores and timestamp. Uploaded files and input values are kept out of new history entries.

## Example inputs

- Download the CSV header template from either tabular page. Use UTF-8 and fill every required field. The kidney page accepts both the template column names and the original dataset abbreviations. Its categorical fields accept the values shown in the form or their 0/1 encoding.
- Upload a PNG or JPEG sample for either image model. Model sizes and class mappings are documented with the checkpoints.

[Checkpoint documentation](saved_models/README.md)

## Configuration and tests

The app binds to localhost by default. For a different database, set `DATABASE_URL` in the environment or copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and set it there. The application has adapters for SQLite, MongoDB, PostgreSQL and MySQL. The checked local configuration uses SQLite.

Run the inference and Streamlit regression tests with the bundled Git LFS checkpoints.

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## License

See the [MIT license](LICENSE) for the copyright and permission notices.
