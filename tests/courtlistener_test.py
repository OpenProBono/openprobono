"""Tests for CourtListener."""

def test_batchupload(capsys) -> None:
    from app.courtlistener import batch_metadata_files


    #_ = create_collection("courtlistener_bulk", EncoderParams(dim=1536), "Opinions from CourtListener bulk data.")

    with capsys.disabled():
        batch_metadata_files()

def test_docketnulls(capsys) -> None:
    import bz2

    import pandas as pd

    def count_null_values_pandas(filename):
        null_counts = {}
        total_rows = 0

        with bz2.open(filename, "rt") as bz_file:
            for chunk in pd.read_csv(bz_file, chunksize=100000, low_memory=False):
                total_rows += len(chunk)
                if "citations" in chunk.columns:
                    print("yes")
                null_counts_chunk = chunk.isna().sum()
                for column, count in null_counts_chunk.items():
                    null_counts[column] = null_counts.get(column, 0) + count

                if total_rows % 1000000 == 0:
                    print(f"Processed {total_rows} rows...")

        print(f"\nTotal rows: {total_rows}")
        print("\nNull/Empty value counts per column:")
        for column, count in null_counts.items():
            percentage = (count / total_rows) * 100
            print(f"{column}: {count} ({percentage:.2f}%)")

    # Usage
    filename = "opinion-clusters-2024-05-06.csv.bz2"
    with capsys.disabled():
        count_null_values_pandas(filename)
