"""Verify that the vision override cannot leak to text lowering, even on failure."""
from types import SimpleNamespace
import unittest
from run_page_pipeline_vision_precision import converter_scope


class PrecisionScope(unittest.TestCase):
    def test_restores_stock_after_success_and_compile_failure(self):
        for fail in [False,True]:
            stock=lambda **kw: kw
            ge=SimpleNamespace(PromptFlashAttention=stock)
            op=SimpleNamespace(_ge_converter=object())
            original=op._ge_converter
            records=[]
            def install(mode):
                self.assertEqual(mode,4)
                op._ge_converter=object()
            try:
                with converter_scope(op,ge,install,records.append):
                    self.assertEqual(ge.PromptFlashAttention(inner_precise=4)['inner_precise'],4)
                    if fail:raise RuntimeError('simulated compile failure')
            except RuntimeError:
                if not fail:raise
            self.assertIs(op._ge_converter,original)
            self.assertIs(ge.PromptFlashAttention,stock)
            self.assertEqual(ge.PromptFlashAttention(inner_precise=1)['inner_precise'],1)
            self.assertEqual(len(records),1)


if __name__=='__main__':unittest.main()
