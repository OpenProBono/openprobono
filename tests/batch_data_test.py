"""Test batch data methods."""

def test_bulkupload() -> None:
    from app.batch_data import upload_cl_completed_batches

    upload_cl_completed_batches()
