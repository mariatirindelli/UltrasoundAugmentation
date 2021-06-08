import unittest


class BaseTest(unittest.TestCase):

    def assetEqualList(self, l1, l2, order_matters=False):
        if len(l1) != len(l2):
            return False

        if order_matters:
            for i in range(len(l1)):
                msg = "element {} not equal - {} != {}".format(i, l1[i], l2[i])
                self.assertEqual(l1[i], l2[i], msg=msg)
            return

        for item in l1:
            msg = "element {} not in l2".format(item)
            self.assertTrue(item in l2, msg=msg)

        for item in l2:
            msg = "element {} not in l1".format(item)
            self.assertTrue(item in l1, msg=msg)


if __name__ == '__main__':
    unittest.main()
