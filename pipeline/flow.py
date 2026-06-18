from prefect import flow, task
import subprocess, sys

@task(retries=2, retry_delay_seconds=30)
def run_ingest():
    print("Ingesting shot data...")
    subprocess.run([sys.executable, "pipeline/ingest.py"], check=True)

@task
def run_clean():
    subprocess.run([sys.executable, "pipeline/clean.py"], check=True)

@task
def run_train():
    subprocess.run([sys.executable, "pipeline/train.py"], check=True)

@flow(name="nba-shot-pipeline")
def nba_pipeline():
    ingest = run_ingest.submit()
    clean = run_clean.submit(wait_for=[ingest])
    train = run_train.submit(wait_for=[clean])
    train.result()  # blocks until the full chain finishes

if __name__ == "__main__":
    nba_pipeline()