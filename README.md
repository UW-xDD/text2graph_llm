# text2graph_llm

An experimental API endpoint to convert text to knowledge graph triplets.

## Usage

### OLVI

1. docker-compose up
2. nohup python3 main.py --run_name olvi --input_file "data/formation_sample.parquet.gzip" --workers 6 &
3. see working.ipynb to calculate expected run time

### CHTC (not working yet)

Note. running ollama on shell works, but using python to access localhost ollama endpoint does not work (`ollama` and `requests` both fails).

1. Get staging drive on CHTC (e.g. /staging/clo36)
2. Put SQLite database in staging drive
3. Set `.env` with `SQLITE_DB_PATH` pointing to the database (e.g.: SQLITE_DB_PATH="data/eval.db")
4. Modify [job file](chtc_job.sub) to suit your needs
5. SSH into CHTC submit node and submit the job file (e.g.: `condor_submit chtc_job.sub`)
