from text2graph.utils import get_eta

if __name__ == "__main__":

    print(
        get_eta(
            eval_db="data/eval.db",
            test_set="data/formation_sample.parquet.gzip",
            run_name="olvi",
            n_workers=3,
        )
    )
