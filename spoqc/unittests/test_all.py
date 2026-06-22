
#In[]
import unittest
import pandas as pd
import dask.dataframe as dd
import builtins

class TestParquetEquality(unittest.TestCase):

    def setUp(self):
        # HQCR data
        self.test_data_pd = pd.read_parquet(
            f'{builtins.TEST_DATA_DIR}/hqcr_output_mask_raw.parquet',
            engine="pyarrow"
        )
        self.ref_data_pd = pd.read_parquet(
            f'{builtins.REF_DATA_DIR}/hqcr_output_mask_raw.parquet',
            engine="pyarrow"
        )

    # ---------- Helper for Dask comparisons ----------
    def assert_dask_equal(self, test_path, ref_path):
        test_dd = dd.read_parquet(test_path, engine="pyarrow")
        ref_dd = dd.read_parquet(ref_path, engine="pyarrow")

        # Memory-friendly hash comparison
        test_hash = test_dd.map_partitions(
            lambda df: pd.util.hash_pandas_object(df, index=True).sum()
        ).compute().sum()

        ref_hash = ref_dd.map_partitions(
            lambda df: pd.util.hash_pandas_object(df, index=True).sum()
        ).compute().sum()

        self.assertEqual(test_hash, ref_hash, f"Mismatch in dataset: {test_path}")

    # ---------- HQCR test ----------
    def test_pandas_dataframes_equal(self):
        pd.testing.assert_frame_equal(
            self.test_data_pd,
            self.ref_data_pd,
            check_dtype=True
        )

    # ---------- HQPR tests ----------
    def test_hqpr_0_output_mask_raw(self):
        self.assert_dask_equal(
            f'{builtins.TEST_DATA_DIR}/hqpr_0_output_mask_raw',
            f'{builtins.REF_DATA_DIR}/hqpr_0_output_mask_raw'
        )

    def test_hqpr_0_output_mask_raw(self):
        self.assert_dask_equal(
            f'{builtins.TEST_DATA_DIR}/hqpr_0_output_mask_raw',
            f'{builtins.REF_DATA_DIR}/hqpr_0_output_mask_raw'
        )

     # ---------- HQTR tests ----------
    def test_hqtr_output_mask_raw(self):
        self.assert_dask_equal(
            f'{builtins.TEST_DATA_DIR}/hqtr_output_mask_raw',
            f'{builtins.REF_DATA_DIR}/hqtr_output_mask_raw'
        )

    def test_hqtr_output_qv_prob(self):
        self.assert_dask_equal(
            f'{builtins.TEST_DATA_DIR}/hqtr_output_qv_prob',
            f'{builtins.REF_DATA_DIR}/hqtr_output_qv_prob'
        )

    def test_hqtr_output_ac_prob(self):
        self.assert_dask_equal(
            f'{builtins.TEST_DATA_DIR}/hqtr_output_ac_prob',
            f'{builtins.REF_DATA_DIR}/hqtr_output_ac_prob'
        )

    def test_hqtr_output_mask_raw(self):
        self.assert_dask_equal(
            f'{builtins.TEST_DATA_DIR}/hqtr_output_mask_raw',
            f'{builtins.REF_DATA_DIR}/hqtr_output_mask_raw'
        )

# %%
