"""检查交换数据中用于回查的事实。"""
import unittest

import numpy as np

from persistence_loss_input import load_record


class ExchangeRecordTests(unittest.TestCase):
    def test_exchange_contract(self):
        record = load_record()
        self.assertTrue(record["tiles"][0]["origin"][1] + len(record["tiles"][0]["values"][0]) - 1 == record["tiles"][1]["origin"][1])
        self.assertTrue(all(a[-1] == b[0] for a,b in zip(record["tiles"][0]["values"],record["tiles"][1]["values"])))
        self.assertTrue(record["tiles"][1]["values"][2][1] > record["tiles"][0]["values"][2][1] and [2,4] not in record["permanent_face_holes"])
        self.assertTrue(record["arrival"][-1] == "R" and len(record["regions"]) == 2)
