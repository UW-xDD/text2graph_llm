# text2graph_llm

An experimental API endpoint to convert text to knowledge graph triplets.

## Usage

1. Get staging drive on CHTC (e.g. /staging/clo36)
2. Put SQLite database in staging drive
3. Set `.env` with `SQLITE_DB_PATH` pointing to the database (e.g.: SQLITE_DB_PATH="data/eval.db")
4. Modify [job file](chtc_job.sub) to suit your needs
5. SSH into CHTC submit node and submit the job file (e.g.: `condor_submit chtc_job.sub`)
