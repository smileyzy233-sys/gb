import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from standard_pipeline import cli as cli_module
from standard_pipeline.comparison import run_collect_firm_year_outputs, run_compare_dummy_consistency
from standard_pipeline.config import PipelineConfig
from standard_pipeline.extract import ExtractSettings, process_text_unit_row
from standard_pipeline.gb_mapping import (
    build_mapping_lookup,
    find_mapping,
    map_row,
    parse_standard_code,
    run_gb_mapping,
)
from standard_pipeline.llm import ApiClientConfig, OpenAICompatibleClient, api_config_from_dict
from standard_pipeline.main_regression import (
    TextUnitSettings,
    build_protected_anchor_matcher,
    compress_table_noise,
    call_stage1_with_retry,
    clean_chapter_text_for_units,
    default_main_regression_paths,
    extract_target_chapter_blocks,
    has_protected_anchor_in_text,
    match_keyword_terms,
    run_aggregate_main,
    run_build_text_units,
    run_route_main,
    run_stage1_screening,
    run_text_unit_noise_audit,
    split_text_units,
)
from standard_pipeline.preprocess import (
    PreprocessSettings,
    parse_report_filename,
)
from standard_pipeline.robustness import (
    default_robustness_paths,
    run_prepare_full_units,
    run_prepare_keyword_units,
    run_prepare_llm_units,
)
from standard_pipeline.schemas import ROUTE_AUDIT_COLUMNS, STAGE1_RELEVANCE_COLUMNS, TEXT_UNIT_EXTRACTION_COLUMNS
from standard_pipeline.vllm_batch import (
    VLLMBatchConfig,
    run_stage1_screening_vllm_batch,
    run_text_unit_extraction_vllm_batch,
    vllm_batch_config_from_dict,
)


class PreprocessTests(unittest.TestCase):
    def test_parse_report_filename(self):
        self.assertEqual(
            parse_report_filename("000001_2024_平安银行_年度报告.txt"),
            {"stock_code": "000001", "year": "2024", "company_name": "平安银行"},
        )
        self.assertEqual(
            parse_report_filename("000001_平安银行_2024_年度报告.txt"),
            {"stock_code": "000001", "year": "2024", "company_name": "平安银行"},
        )

class GbMappingTests(unittest.TestCase):
    def test_parse_standard_code(self):
        self.assertEqual(parse_standard_code("GB/T 19001-2016"), ("GB/T19001", "2016"))
        self.assertEqual(parse_standard_code("GB 3836.1-2021"), ("GB3836.1", "2021"))

    def test_missing_entity_year_uses_version_active_at_report_cutoff(self):
        mapping = pd.DataFrame(
            [
                {
                    "标准号": "GB/T 13471-2025",
                    "国际标准编号": "ISO/TS 50044:2019",
                    "采标类型": "非等效采用",
                    "生效日期": "2026-05-01",
                    "失效日期": None,
                    "当前状态": "即将实施",
                    "时间数据质量": "NOT_EFFECTIVE_AT_SNAPSHOT",
                },
                {
                    "标准号": "GB/T 13471-2008",
                    "国际标准编号": None,
                    "采标类型": None,
                    "生效日期": "2009-05-01",
                    "失效日期": "2026-05-01",
                    "当前状态": "现行",
                    "时间数据质量": "OK",
                },
            ]
        )
        lookup = build_mapping_lookup(mapping)

        historical = find_mapping("GB/T 13471", lookup, "2020")
        future = find_mapping("GB/T 13471", lookup, "2026")
        mapped = map_row(
            pd.Series({"entity": "GB/T 13471", "type": "TYPE_B", "status": "ADOPTED", "year": "2020"}),
            lookup,
        )

        self.assertIsNotNone(historical)
        self.assertEqual(historical.original_code, "GB/T 13471-2008")
        self.assertIsNotNone(future)
        self.assertEqual(future.original_code, "GB/T 13471-2025")
        self.assertEqual(mapped[2], 0)

    def test_explicit_version_never_falls_back_to_another_year(self):
        mapping = pd.DataFrame(
            [
                {"标准号": "GB/T 19001-2016", "国际标准编号": "ISO 9001:2015", "采标类型": "等同采用"},
                {"标准号": "GB/T 19001-2024", "国际标准编号": "ISO 9001:2024", "采标类型": "等同采用"},
            ]
        )
        lookup = build_mapping_lookup(mapping)

        self.assertIsNone(find_mapping("GB/T 19001-2020", lookup, "2020"))

    def test_explicit_version_after_report_cutoff_is_rejected(self):
        mapping = pd.DataFrame(
            [
                {
                    "标准号": "GB/T 17650.2-2021",
                    "国际标准编号": "IEC 60754-2:2019",
                    "采标类型": "等同采用",
                    "生效日期": "2021-11-01",
                    "当前状态": "现行",
                    "时间数据质量": "OK",
                }
            ]
        )
        lookup = build_mapping_lookup(mapping)

        self.assertIsNone(find_mapping("GB/T 17650.2-2021", lookup, "2020"))
        self.assertIsNotNone(find_mapping("GB/T 17650.2-2021", lookup, "2021"))

    def test_deprecated_version_with_known_expiry_is_used_historically(self):
        mapping = pd.DataFrame(
            [
                {
                    "标准号": "GB/T 17650.2-2021",
                    "国际标准编号": "IEC 60754-2:2019",
                    "采标类型": "等同采用",
                    "生效日期": "2021-11-01",
                    "当前状态": "现行",
                    "时间数据质量": "OK",
                },
                {
                    "标准号": "GB/T 17650.2-1998",
                    "国际标准编号": "IEC 60754-2:1991",
                    "采标类型": "等同采用",
                    "生效日期": "1999-10-01",
                    "失效日期": "2021-11-01",
                    "当前状态": "废止",
                    "时间数据质量": "OK",
                },
            ]
        )
        lookup = build_mapping_lookup(mapping)

        historical = find_mapping("GB/T 17650.2", lookup, "2020")

        self.assertIsNotNone(historical)
        self.assertEqual(historical.international_standard, "IEC 60754-2:1991")

    def test_run_gb_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_csv = tmp_path / "final.csv"
            mapping_csv = tmp_path / "GB_dict.csv"
            output_csv = tmp_path / "mapped.csv"

            pd.DataFrame(
                [
                    {
                        "text_unit_id": "000001_2024_00001",
                        "stock_code": "000001",
                        "company_name": "TEST",
                        "year": "2024",
                        "entity": "GB/T 19001-2016",
                        "type": "TYPE_B",
                        "status": "ADOPTED",
                        "evidence": "通过GB/T 19001认证",
                    }
                ]
            ).to_csv(input_csv, index=False, encoding="utf-8-sig")

            pd.DataFrame(
                [
                    {
                        "标准号": "GB/T 19001-2016",
                        "国际标准编号": "ISO 9001",
                        "采标类型": "等同采用",
                    }
                ]
            ).to_csv(mapping_csv, index=False, encoding="utf-8-sig")

            result = run_gb_mapping(input_csv, mapping_csv, output_csv)
            written = pd.read_csv(output_csv, dtype=str)

        self.assertEqual(result.loc[0, "国际标准"], "ISO 9001")
        self.assertEqual(result.loc[0, "output"], 1)
        self.assertIn("text_unit_id", written.columns)
        self.assertEqual(written.loc[0, "text_unit_id"], "000001_2024_00001")


class LlmConfigTests(unittest.TestCase):
    def test_api_config_reads_optional_max_tokens_and_extra_body(self):
        config = api_config_from_dict(
            {
                "model": "stage-model",
                "temperature": 0.0,
                "json_response_format": False,
                "max_tokens": 128,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            },
            api_key="EMPTY",
        )
        legacy_config = api_config_from_dict(
            {"model": "legacy-model", "extra_body": "ignored"},
            api_key="EMPTY",
        )

        self.assertEqual(config.max_tokens, 128)
        self.assertFalse(config.json_response_format)
        self.assertEqual(config.extra_body, {"chat_template_kwargs": {"enable_thinking": False}})
        self.assertIsNone(legacy_config.max_tokens)
        self.assertIsNone(legacy_config.extra_body)

    def test_openai_compatible_client_posts_chat_completions_payload(self):
        class DummyResponse:
            status_code = 200
            text = '{"ok": true}'

            def json(self):
                return {"choices": [{"message": {"content": '{"answer": "ok"}'}}]}

        config = ApiClientConfig(
            api_key="EMPTY",
            base_url="http://127.0.0.1:18000/v1/",
            model="qwen3.5-9b",
            temperature=0.0,
            json_response_format=True,
            max_tokens=128,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

        with patch("standard_pipeline.llm.requests.post", return_value=DummyResponse()) as post:
            content = OpenAICompatibleClient(config).complete_json("system", "user")

        self.assertEqual(content, '{"answer": "ok"}')
        post.assert_called_once()
        _, kwargs = post.call_args
        self.assertEqual(post.call_args.args[0], "http://127.0.0.1:18000/v1/chat/completions")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer EMPTY")
        self.assertEqual(kwargs["headers"]["Content-Type"], "application/json")
        self.assertEqual(kwargs["timeout"], 300)
        self.assertEqual(kwargs["json"]["model"], "qwen3.5-9b")
        self.assertEqual(kwargs["json"]["max_tokens"], 128)
        self.assertEqual(kwargs["json"]["response_format"], {"type": "json_object"})
        self.assertEqual(kwargs["json"]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(
            kwargs["json"]["messages"],
            [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}],
        )

    def test_openai_compatible_client_raises_runtime_error_for_non_2xx(self):
        class DummyResponse:
            status_code = 502
            text = "bad gateway " + ("x" * 2100)

            def json(self):
                return {}

        config = ApiClientConfig(
            api_key="EMPTY",
            base_url="http://127.0.0.1:18000/v1",
            model="qwen3.5-9b",
        )

        with patch("standard_pipeline.llm.requests.post", return_value=DummyResponse()):
            with self.assertRaises(RuntimeError) as context:
                OpenAICompatibleClient(config).complete_json("system", "user")

        message = str(context.exception)
        self.assertIn("status_code=502", message)
        self.assertIn("url=http://127.0.0.1:18000/v1/chat/completions", message)
        self.assertIn("model=qwen3.5-9b", message)
        self.assertIn("bad gateway", message)

    def test_stage_extract_settings_use_stage_values_and_cli_overrides(self):
        config = PipelineConfig(
            Path("."),
            {
                "extract": {"workers": 2, "batch_size": 10, "max_retries": 1},
                "stage1": {
                    "workers": 8,
                    "batch_size": 50,
                    "max_retries": 3,
                    "retry_min_seconds": 2,
                    "retry_max_seconds": 10,
                },
            },
        )

        settings = cli_module.stage_extract_settings(config, "stage1", workers=None, batch_size=None)
        overridden = cli_module.stage_extract_settings(config, "stage1", workers=4, batch_size=25)

        self.assertEqual(settings.workers, 8)
        self.assertEqual(settings.batch_size, 50)
        self.assertEqual(settings.max_retries, 3)
        self.assertEqual(overridden.workers, 4)
        self.assertEqual(overridden.batch_size, 25)

    def test_stage_provider_prefers_cli_then_stage_then_extract(self):
        config = PipelineConfig(
            Path("."),
            {
                "extract": {"provider": "api"},
                "stage1": {"provider": "local"},
                "stage2": {},
            },
        )

        self.assertEqual(cli_module.stage_provider(config, "stage1"), "local")
        self.assertEqual(cli_module.stage_provider(config, "stage1", "api"), "api")
        self.assertEqual(cli_module.stage_provider(config, "stage2"), "api")

    def test_vllm_batch_config_uses_cli_env_then_config_path(self):
        with patch.dict(
            "os.environ",
            {"LOCAL_MODEL_PATH": "env-model", "CUSTOM_MODEL_PATH": "custom-env-model"},
            clear=False,
        ):
            from_env = vllm_batch_config_from_dict(
                {
                    "model_path_env": "CUSTOM_MODEL_PATH",
                    "model_path": "config-model",
                    "chunk_size": 32,
                    "lora_path": "",
                }
            )
            from_cli = vllm_batch_config_from_dict(
                {"model_path_env": "CUSTOM_MODEL_PATH", "model_path": "config-model"},
                model_path="cli-model",
            )

        self.assertEqual(from_env.model_path, "custom-env-model")
        self.assertEqual(from_env.chunk_size, 32)
        self.assertIsNone(from_env.lora_path)
        self.assertEqual(from_cli.model_path, "cli-model")

    def test_vllm_batch_config_requires_model_path(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError) as context:
                vllm_batch_config_from_dict({})

        self.assertIn("Missing vLLM batch model path", str(context.exception))

    def test_provider_model_label_supports_vllm_batch(self):
        config = PipelineConfig(
            Path("."),
            {
                "stage1": {
                    "vllm_batch": {
                        "model_path_env": "CUSTOM_MODEL_PATH",
                        "model_path": "config-model",
                    }
                }
            },
        )

        with patch.dict("os.environ", {"CUSTOM_MODEL_PATH": "env-model"}, clear=False):
            label = cli_module.provider_model_label(config, "vllm_batch", stage_name="stage1")
            override_label = cli_module.provider_model_label(
                config,
                "vllm_batch",
                model_path="cli-model",
                stage_name="stage1",
            )

        self.assertEqual(label, "vllm_batch:env-model")
        self.assertEqual(override_label, "vllm_batch:cli-model")

    def test_build_stage_client_uses_stage_api_config(self):
        class DummyOpenAIClient:
            def __init__(self, config):
                self.config = config

        config = PipelineConfig(
            Path("."),
            {
                "extract": {
                    "api": {"model": "extract-model", "base_url": "http://extract", "max_tokens": 384}
                },
                "stage1": {
                    "api": {"model": "stage1-model", "base_url": "http://stage1", "max_tokens": 128}
                },
                "stage2": {
                    "api": {"model": "stage2-model", "base_url": "http://stage2", "max_tokens": 384}
                },
            },
        )

        with patch.object(cli_module, "OpenAICompatibleClient", DummyOpenAIClient):
            stage1_client = cli_module.build_stage_client(config, "stage1", "api", api_key="EMPTY")
            stage2_client = cli_module.build_stage_client(config, "stage2", "api", api_key="EMPTY")

        self.assertEqual(stage1_client.config.model, "stage1-model")
        self.assertEqual(stage1_client.config.base_url, "http://stage1")
        self.assertEqual(stage1_client.config.max_tokens, 128)
        self.assertEqual(stage2_client.config.model, "stage2-model")
        self.assertEqual(stage2_client.config.max_tokens, 384)

    def test_build_stage_client_rejects_vllm_batch(self):
        config = PipelineConfig(Path("."), {})

        with self.assertRaises(ValueError) as context:
            cli_module.build_stage_client(config, "stage1", "vllm_batch")

        self.assertIn("does not use ChatClient", str(context.exception))


class MainRegressionTests(unittest.TestCase):
    def test_text_unit_id_is_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            report_dir = tmp_path / "reports"
            report_dir.mkdir()
            report = report_dir / "000001_2024_平安银行_年度报告.txt"
            report.write_text(
                "第一节 经营情况\n公司通过 ISO 9001 认证。产品符合 GB/T 19001 要求。\n"
                "第二节 财务报告\n本节不应进入目标章节。",
                encoding="utf-8",
            )
            settings = PreprocessSettings(min_chapter_chars=1, target_chapter_keywords=("经营情况",))
            unit_settings = TextUnitSettings(min_chars=10, max_chars=80)
            first = run_build_text_units(report_dir, tmp_path / "first.csv", settings, unit_settings)
            second = run_build_text_units(report_dir, tmp_path / "second.csv", settings, unit_settings)

        self.assertEqual(first["text_unit_id"].tolist(), second["text_unit_id"].tolist())
        self.assertEqual(first.loc[0, "text_unit_id"], "000001_2024_00001")

    def test_build_text_units_accepts_explicit_report_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            report_dir = tmp_path / "reports"
            report_dir.mkdir()
            first_report = report_dir / "000001_2024_A_年度报告.txt"
            second_report = report_dir / "000002_2024_B_年度报告.txt"
            first_report.write_text("第一节 经营情况\nA 公司通过 ISO 9001 认证。", encoding="utf-8")
            second_report.write_text("第一节 经营情况\nB 公司通过 ISO 14001 认证。", encoding="utf-8")

            result = run_build_text_units(
                report_dir,
                tmp_path / "sample.csv",
                PreprocessSettings(min_chapter_chars=1, target_chapter_keywords=("经营情况",)),
                TextUnitSettings(min_chars=10, max_chars=80),
                input_files=[second_report],
            )

        self.assertEqual(result["stock_code"].tolist(), ["000002"])
        self.assertEqual(result["source_file"].tolist(), [second_report.name])

    def test_build_text_units_uses_supplemental_protected_terms(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            report_dir = tmp_path / "reports"
            report_dir.mkdir()
            report = report_dir / "000001_2024_A_年度报告.txt"
            report.write_text(
                "第一节 经营情况\n产品\nAX1\nJEDEC JESD47\n版本\n123456",
                encoding="utf-8",
            )

            result = run_build_text_units(
                report_dir,
                tmp_path / "units.csv",
                PreprocessSettings(min_chapter_chars=1, target_chapter_keywords=("经营情况",)),
                TextUnitSettings(min_chars=1, max_chars=80),
                protected_anchor_terms=["JEDEC"],
            )

        self.assertEqual(len(result), 1)
        self.assertIn("JEDEC JESD47", result.loc[0, "text"])

    def test_chapter_filtering_selects_target_chapter(self):
        settings = PreprocessSettings(min_chapter_chars=1, target_chapter_keywords=("经营情况",))
        raw = "第一节 其他事项\n这里没有标准。\n第二节 经营情况\n公司通过 ISO 9001 认证。\n第三节 财务报告\n利润增长。"
        blocks = extract_target_chapter_blocks(raw, settings)

        self.assertEqual(len(blocks), 1)
        self.assertIn("经营情况", blocks[0][0])
        self.assertIn("ISO 9001", blocks[0][1])
        self.assertNotIn("利润增长", blocks[0][1])

    def test_chapter_filtering_skips_toc_and_cross_reference_titles(self):
        settings = PreprocessSettings(min_chapter_chars=1, target_chapter_keywords=("管理层讨论",))
        raw = (
            "重要提示\n"
            "第三节 管理层讨论与分析中关于公司未来发展的展望部分描述了风险。\n"
            "目录\n"
            "第三节 管理层讨论与分析........................................ 23\n"
            "3.1 总体经营情况................................................ 24\n"
            "第三节\n"
            "管理层讨论与分析\n"
            "公司通过 ISO 9001 认证,并持续满足 GB/T 19001 要求。\n"
            "第四节 公司治理\n"
            "公司治理内容不应进入。"
        )

        blocks = extract_target_chapter_blocks(raw, settings)

        self.assertEqual(len(blocks), 1)
        self.assertIn("管理层讨论", blocks[0][0])
        self.assertIn("ISO 9001", blocks[0][1])
        self.assertNotIn("目录", blocks[0][1])
        self.assertNotIn("描述了风险", blocks[0][1])
        self.assertNotIn("公司治理内容", blocks[0][1])

    def test_chapter_filtering_keeps_cross_reference_continuation_and_ignores_repeated_header(self):
        settings = PreprocessSettings(min_chapter_chars=1, target_chapter_keywords=("董事会报告",))
        raw = (
            "第四节 董事会报告\n"
            "本节开头。\n"
            "有关项目详见本报告“第四节董事会报告”之“项目投资情况”。\n"
            "交叉引用后的 ISO 9001 证据必须保留。\n"
            "公司名称 2021 年度报告\n"
            "第四节 董事会报告\n"
            "第二页的 GB/T 19001 证据也必须保留。\n"
            "第五节 公司治理\n"
            "下一章不应进入。"
        )

        blocks = extract_target_chapter_blocks(raw, settings)

        self.assertEqual(len(blocks), 1)
        self.assertIn("交叉引用后的 ISO 9001", blocks[0][1])
        self.assertIn("第二页的 GB/T 19001", blocks[0][1])
        self.assertNotIn("下一章不应进入", blocks[0][1])

    def test_clean_chapter_text_removes_headers_and_merges_short_table_lines(self):
        text = (
            "35\n"
            "平安银行股份有限公司\n"
            "2024 年年度报告\n"
            "污染\n"
            "苯\n"
            "有组\n"
            "织排\n"
            "放\n"
            "未超\n"
            "标\n"
            "公司通过 ISO 9001 认证。\n"
        )

        cleaned = clean_chapter_text_for_units(text, year="2024")

        self.assertNotIn("35\n平安银行股份有限公司", cleaned)
        self.assertNotIn("2024 年年度报告", cleaned)
        self.assertIn("污染苯有组织排放未超标", cleaned)
        self.assertIn("公司通过 ISO 9001 认证。", cleaned)

    def test_table_compression_drops_numeric_blocks_but_keeps_protected_context(self):
        text = (
            "普通正文应保留。\n\n"
            "项目\n2021\n1,234.50\n18.5%\n2021-12-31\n9,876\n\n"
            "认证项目\nISO 9001\n证书编号\n123456\n2024-12-31\n88.8%\n"
            "结尾正文。"
        )

        filtered = compress_table_noise(text)

        self.assertIn("普通正文应保留", filtered)
        self.assertNotIn("1,234.50", filtered)
        self.assertNotIn("9,876", filtered)
        self.assertIn("ISO 9001", filtered)
        self.assertIn("证书编号", filtered)
        self.assertIn("123456", filtered)
        self.assertIn("结尾正文", filtered)

    def test_table_compression_protects_registration_lists(self):
        text = (
            "产品名称\n吉械注准\n20192400123\nII类\n"
            "用于体外定量检测\n2020.03.13-2024.09.09\n延续注册"
        )

        filtered = compress_table_noise(text)

        self.assertIn("吉械注准", filtered)
        self.assertIn("20192400123", filtered)
        self.assertIn("延续注册", filtered)

    def test_table_compression_protects_traditional_standard_terms(self):
        text = "年度\n2021\n國際財務報告準則\nIFRS 9\n100%\n報告期"

        filtered = compress_table_noise(text)

        self.assertIn("國際財務報告準則", filtered)
        self.assertIn("IFRS 9", filtered)

    def test_split_table_anchor_is_detected(self):
        self.assertTrue(has_protected_anchor_in_text("项目\n认\n证\n编号\n12345"))

    def test_table_compression_uses_supplemental_keyword_anchor(self):
        text = "产品\nAX1\nJEDEC JESD47\n版本\n123456\n2024-12-31"
        protected_anchor_matcher = build_protected_anchor_matcher(["JEDEC"])

        self.assertNotIn("JEDEC", compress_table_noise(text))
        filtered = compress_table_noise(text, protected_anchor_matcher)

        self.assertIn("JEDEC JESD47", filtered)
        self.assertIn("123456", filtered)

    def test_supplemental_short_ascii_keyword_keeps_word_boundaries(self):
        protected_anchor_matcher = build_protected_anchor_matcher(["UN"])

        self.assertTrue(has_protected_anchor_in_text("项目\nUN\n12345", protected_anchor_matcher))
        self.assertFalse(has_protected_anchor_in_text("项目\nJUNE\n12345", protected_anchor_matcher))

    def test_text_unit_noise_audit_is_dry_run_and_has_expected_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "01_text_units_2021.csv"
            audit = tmp_path / "01_text_units_2021_audit.csv"
            original = pd.DataFrame(
                [
                    {
                        "text_unit_id": "000001_2021_00001",
                        "stock_code": "1",
                        "company_name": "平安银行",
                        "year": "2021",
                        "source_file": "report.txt",
                        "chapter_title": "管理层讨论与分析",
                        "unit_order": 1,
                        "text": "25\n项目\n2021\n1,234\n18.5%\n9,876",
                    }
                ]
            )
            original.to_csv(source, index=False, encoding="utf-8-sig")
            before = source.read_bytes()

            count = run_text_unit_noise_audit(source, audit, chunksize=1)
            result = pd.read_csv(audit)

            self.assertEqual(source.read_bytes(), before)

        self.assertEqual(count, 1)
        self.assertEqual(
            result.columns.tolist(),
            [
                "text_unit_id",
                "company_name",
                "unit_order",
                "table_noise_score",
                "has_protected_anchor",
                "raw_text_len",
                "filtered_text_len",
                "removed_ratio",
                "raw_preview",
                "filtered_preview",
            ],
        )
        self.assertGreater(result.loc[0, "removed_ratio"], 0)

    def test_long_paragraph_split_keeps_text(self):
        text = "公司通过 ISO 9001 认证。" * 80 + "持续符合 CE 准入要求。" * 80
        units = split_text_units(text, TextUnitSettings(min_chars=50, max_chars=120))
        joined = "".join(unit.replace("\n", "") for unit in units)

        self.assertGreater(len(units), 1)
        self.assertEqual(joined, text)

    def test_keyword_matching_detects_standard_terms(self):
        matched = match_keyword_terms(
            "公司通过 ISO 9001 和 GB/T 19001 认证，并取得 CE 认证。DISORDER 不应触发 ISO。",
            ["GB/T", "ISO", "CE"],
        )

        self.assertIn("ISO", matched)
        self.assertIn("GB/T", matched)
        self.assertIn("CE", matched)

    def test_route_main_uses_keyword_or_stage1_union(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            text_units = tmp_path / "text_units.csv"
            keyword = tmp_path / "keyword.csv"
            stage1 = tmp_path / "stage1.csv"
            output = tmp_path / "stage2.csv"
            base_rows = [
                {
                    "text_unit_id": f"id{i}",
                    "stock_code": "000001",
                    "company_name": "TEST",
                    "year": "2024",
                    "source_file": "r.txt",
                    "chapter_title": "经营情况",
                    "unit_order": i,
                    "text": f"text {i}",
                }
                for i in range(1, 5)
            ]
            pd.DataFrame(base_rows).to_csv(text_units, index=False, encoding="utf-8-sig")
            pd.DataFrame(
                [
                    {"text_unit_id": "id1", "keyword_candidate": True, "matched_terms": "ISO"},
                    {"text_unit_id": "id2", "keyword_candidate": False, "matched_terms": ""},
                    {"text_unit_id": "id3", "keyword_candidate": False, "matched_terms": ""},
                    {"text_unit_id": "id4", "keyword_candidate": False, "matched_terms": ""},
                ]
            ).to_csv(keyword, index=False, encoding="utf-8-sig")
            pd.DataFrame(
                [
                    {"text_unit_id": "id1", "relevance": "unrelated", "confidence_score": 0.8, "reason": "", "stage1_status": "OK", "stage1_error": ""},
                    {"text_unit_id": "id2", "relevance": "related", "confidence_score": 0.9, "reason": "", "stage1_status": "OK", "stage1_error": ""},
                    {"text_unit_id": "id3", "relevance": "uncertain", "confidence_score": 0.5, "reason": "", "stage1_status": "OK", "stage1_error": ""},
                    {"text_unit_id": "id4", "relevance": "unrelated", "confidence_score": 0.9, "reason": "", "stage1_status": "OK", "stage1_error": ""},
                ]
            ).to_csv(stage1, index=False, encoding="utf-8-sig")

            routed = run_route_main(text_units, keyword, stage1, output)

        self.assertEqual(routed["text_unit_id"].tolist(), ["id1", "id2", "id3"])
        self.assertEqual(routed.columns.tolist(), ROUTE_AUDIT_COLUMNS)
        self.assertEqual(routed["route_reason"].tolist(), ["keyword", "stage1_related", "stage1_uncertain"])

    def test_stage1_screening_skips_keyword_candidates(self):
        class CountingClient:
            def __init__(self):
                self.calls = 0

            def complete_json(self, system_prompt, user_content):
                self.calls += 1
                return '{"text_unit_id": "id2", "relevance": "unrelated", "confidence_score": 0.9, "reason": ""}'

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            text_units = tmp_path / "text_units.csv"
            keyword = tmp_path / "keyword.csv"
            output = tmp_path / "stage1.csv"
            pd.DataFrame(
                [
                    {
                        "text_unit_id": "id1",
                        "stock_code": "000001",
                        "company_name": "TEST",
                        "year": "2024",
                        "source_file": "r.txt",
                        "chapter_title": "经营情况",
                        "unit_order": 1,
                        "text": "公司通过 ISO 9001 认证。",
                    },
                    {
                        "text_unit_id": "id2",
                        "stock_code": "000001",
                        "company_name": "TEST",
                        "year": "2024",
                        "source_file": "r.txt",
                        "chapter_title": "经营情况",
                        "unit_order": 2,
                        "text": "收入增长。",
                    },
                ]
            ).to_csv(text_units, index=False, encoding="utf-8-sig")
            pd.DataFrame(
                [
                    {"text_unit_id": "id1", "keyword_candidate": True, "matched_terms": "ISO"},
                    {"text_unit_id": "id2", "keyword_candidate": False, "matched_terms": ""},
                ]
            ).to_csv(keyword, index=False, encoding="utf-8-sig")

            client = CountingClient()
            result = run_stage1_screening(
                text_units,
                output,
                client,
                "prompt",
                ExtractSettings(workers=1, max_retries=3),
                keyword_features_csv=keyword,
            )

        self.assertEqual(client.calls, 1)
        self.assertEqual(result["text_unit_id"].tolist(), ["id1", "id2"])
        self.assertEqual(result.loc[0, "stage1_status"], "SKIPPED_KEYWORD")
        self.assertEqual(result.loc[1, "stage1_status"], "OK")

    def test_stage1_vllm_batch_writes_error_rows_and_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            text_units = tmp_path / "text_units.csv"
            keyword = tmp_path / "keyword.csv"
            output = tmp_path / "stage1.csv"
            raw_log = tmp_path / "raw_failures.jsonl"
            pd.DataFrame(
                [
                    {
                        "text_unit_id": "id1",
                        "stock_code": "000001",
                        "company_name": "TEST",
                        "year": "2024",
                        "source_file": "r.txt",
                        "chapter_title": "c",
                        "unit_order": 1,
                        "text": "keyword text",
                    },
                    {
                        "text_unit_id": "id2",
                        "stock_code": "000001",
                        "company_name": "TEST",
                        "year": "2024",
                        "source_file": "r.txt",
                        "chapter_title": "c",
                        "unit_order": 2,
                        "text": "needs llm",
                    },
                ]
            ).to_csv(text_units, index=False, encoding="utf-8-sig")
            pd.DataFrame(
                [
                    {"text_unit_id": "id1", "keyword_candidate": True, "matched_terms": "ISO"},
                    {"text_unit_id": "id2", "keyword_candidate": False, "matched_terms": ""},
                ]
            ).to_csv(keyword, index=False, encoding="utf-8-sig")

            with patch("standard_pipeline.vllm_batch.build_vllm_engine", return_value=(object(), object())):
                with patch("standard_pipeline.vllm_batch.generate_raw_batch", return_value=["not json"]):
                    result = run_stage1_screening_vllm_batch(
                        text_units,
                        output,
                        "prompt",
                        VLLMBatchConfig(model_path="model", chunk_size=8),
                        keyword_features_csv=keyword,
                        raw_failure_log=raw_log,
                    )

            written = pd.read_csv(output, dtype=str)
            raw_log_exists = raw_log.exists()

        self.assertEqual(list(result.columns), STAGE1_RELEVANCE_COLUMNS)
        self.assertEqual(list(written.columns), STAGE1_RELEVANCE_COLUMNS)
        self.assertEqual(result["text_unit_id"].tolist(), ["id1", "id2"])
        self.assertEqual(result.loc[0, "stage1_status"], "SKIPPED_KEYWORD")
        self.assertEqual(result.loc[1, "stage1_status"], "ERROR")
        self.assertTrue(raw_log_exists)

    def test_stage1_vllm_batch_resumes_from_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            text_units = tmp_path / "text_units.csv"
            output = tmp_path / "stage1.csv"
            unit_rows = [
                {
                    "text_unit_id": text_unit_id,
                    "stock_code": "000001",
                    "company_name": "TEST",
                    "year": "2024",
                    "source_file": "r.txt",
                    "chapter_title": "c",
                    "unit_order": order,
                    "text": f"text {order}",
                }
                for order, text_unit_id in enumerate(("id1", "id2"), start=1)
            ]
            pd.DataFrame(unit_rows).to_csv(text_units, index=False, encoding="utf-8-sig")
            pd.DataFrame(
                [
                    {
                        "text_unit_id": "id1",
                        "relevance": "unrelated",
                        "confidence_score": 0.9,
                        "reason": "checkpoint",
                        "stage1_status": "OK",
                        "stage1_error": "",
                    }
                ],
                columns=STAGE1_RELEVANCE_COLUMNS,
            ).to_csv(output, index=False, encoding="utf-8-sig")

            def fake_generate(llm, tokenizer, rows, system_prompt, user_content_builder, config):
                self.assertEqual([row["text_unit_id"] for row in rows], ["id2"])
                return ['{"relevance":"related","confidence_score":0.8,"reason":"new"}']

            with patch("standard_pipeline.vllm_batch.build_vllm_engine", return_value=(object(), object())):
                with patch("standard_pipeline.vllm_batch.generate_raw_batch", side_effect=fake_generate):
                    result = run_stage1_screening_vllm_batch(
                        text_units,
                        output,
                        "prompt",
                        VLLMBatchConfig(model_path="model", chunk_size=8),
                        resume=True,
                    )

        self.assertEqual(result["text_unit_id"].tolist(), ["id1", "id2"])
        self.assertEqual(result["reason"].tolist(), ["checkpoint", "new"])

    def test_stage1_json_parse_failure_retries_once_without_sleep_and_logs_raw(self):
        class BadJsonClient:
            def __init__(self):
                self.calls = 0

            def complete_json(self, system_prompt, user_content):
                self.calls += 1
                return "not json"

        row = pd.Series(
            {
                "text_unit_id": "id1",
                "stock_code": "000001",
                "company_name": "TEST",
                "year": "2024",
                "text": "收入增长。",
            }
        )
        client = BadJsonClient()

        with tempfile.TemporaryDirectory() as tmp:
            raw_log = Path(tmp) / "stage1_raw_failures.jsonl"
            with patch("standard_pipeline.main_regression.time.sleep") as sleep:
                with self.assertRaises(RuntimeError) as context:
                    call_stage1_with_retry(
                        client,
                        "prompt",
                        row,
                        ExtractSettings(workers=1, max_retries=3),
                        raw_failure_log=raw_log,
                    )

            records = [json.loads(line) for line in raw_log.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(client.calls, 2)
        sleep.assert_not_called()
        self.assertIn("Stage1 JSON parse failed after 2 attempts", str(context.exception))
        self.assertEqual([record["attempt"] for record in records], [1, 2])
        self.assertEqual(records[0]["text_unit_id"], "id1")
        self.assertEqual(records[0]["raw_response"], "not json")

    def test_stage1_json_parse_failure_repairs_broken_reason_field(self):
        class BrokenReasonClient:
            def __init__(self):
                self.calls = 0

            def complete_json(self, system_prompt, user_content):
                self.calls += 1
                return (
                    '{\n'
                    '  "text_unit_id": "id1",\n'
                    '  "relevance": "related",\n'
                    '  "confidence_score": 0.85,\n'
                    '  "reason": "mentions "quoted term" and padding\t\t"\n'
                    '}'
                )

        row = pd.Series(
            {
                "text_unit_id": "id1",
                "stock_code": "000001",
                "company_name": "TEST",
                "year": "2024",
                "text": "Company passed ISO 9001 certification.",
            }
        )
        client = BrokenReasonClient()

        with tempfile.TemporaryDirectory() as tmp:
            raw_log = Path(tmp) / "stage1_raw_failures.jsonl"
            result = call_stage1_with_retry(
                client,
                "prompt",
                row,
                ExtractSettings(workers=1, max_retries=3),
                raw_failure_log=raw_log,
            )
            records = [json.loads(line) for line in raw_log.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(client.calls, 1)
        self.assertEqual(result["stage1_status"], "OK")
        self.assertEqual(result["relevance"], "related")
        self.assertEqual(result["confidence_score"], 0.85)
        self.assertIn("quoted term", result["reason"])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["text_unit_id"], "id1")

    def test_stage2_empty_extraction_outputs_type_d_no(self):
        class EmptyClient:
            def complete_json(self, system_prompt, user_content):
                return '{"standards": []}'

        row = pd.Series(
            {
                "text_unit_id": "id1",
                "stock_code": "000001",
                "company_name": "TEST",
                "year": "2024",
                "text": "这里没有相关标准。",
            }
        )
        result = process_text_unit_row(row, EmptyClient(), "prompt", ExtractSettings(max_retries=1, workers=1))

        self.assertTrue(result.success)
        self.assertEqual(result.rows[0]["entity"], "无")
        self.assertEqual(result.rows[0]["type"], "TYPE_D")
        self.assertEqual(result.rows[0]["status"], "NO")

    def test_stage2_vllm_batch_writes_log_and_resumes(self):
        def fake_generate(llm, tokenizer, rows, system_prompt, user_content_builder, config):
            outputs = []
            for row in rows:
                user_content = user_content_builder(row)
                self.assertIn(str(row["text_unit_id"]), user_content)
                if row["text_unit_id"] == "id1":
                    outputs.append(
                        '{"standards": [{"entity": "ISO 9001", "type": "TYPE_A", '
                        '"status": "ADOPTED", "evidence": "passed"}]}'
                    )
                else:
                    outputs.append("not json")
            return outputs

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_csv = tmp_path / "stage2_input.csv"
            output = tmp_path / "stage2_output.csv"
            log_file = tmp_path / "stage2_vllm_batch.log"
            failure_queue = tmp_path / "stage2_failures.jsonl"
            pd.DataFrame(
                [
                    {
                        "text_unit_id": "id1",
                        "stock_code": "000001",
                        "company_name": "TEST",
                        "year": "2024",
                        "source_file": "r.txt",
                        "chapter_title": "c",
                        "unit_order": 1,
                        "text": "standard text",
                    },
                    {
                        "text_unit_id": "id2",
                        "stock_code": "000001",
                        "company_name": "TEST",
                        "year": "2024",
                        "source_file": "r.txt",
                        "chapter_title": "c",
                        "unit_order": 2,
                        "text": "broken output text",
                    },
                ]
            ).to_csv(input_csv, index=False, encoding="utf-8-sig")

            with patch("standard_pipeline.vllm_batch.build_vllm_engine", return_value=(object(), object())):
                with patch("standard_pipeline.vllm_batch.generate_raw_batch", side_effect=fake_generate):
                    result = run_text_unit_extraction_vllm_batch(
                        input_csv,
                        output,
                        log_file,
                        failure_queue,
                        "prompt",
                        VLLMBatchConfig(model_path="model", chunk_size=1),
                        resume=False,
                    )

            with patch("standard_pipeline.vllm_batch.build_vllm_engine", return_value=(object(), object())):
                with patch(
                    "standard_pipeline.vllm_batch.generate_raw_batch",
                    return_value=['{"standards": []}'],
                ):
                    resumed = run_text_unit_extraction_vllm_batch(
                        input_csv,
                        output,
                        log_file,
                        failure_queue,
                        "prompt",
                        VLLMBatchConfig(model_path="model", chunk_size=1),
                        resume=True,
                    )

            log_lines = log_file.read_text(encoding="utf-8").splitlines()

        self.assertEqual(list(result.columns), TEXT_UNIT_EXTRACTION_COLUMNS)
        self.assertEqual(log_lines, ["id1", "id2"])
        self.assertIn("ISO 9001", result["entity"].tolist())
        self.assertNotIn("ERROR", result["entity"].tolist())
        self.assertEqual(set(resumed["text_unit_id"]), {"id1", "id2"})

    def test_aggregate_main_dummy_is_max_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            text_units = tmp_path / "text_units.csv"
            mapped = tmp_path / "mapped.csv"
            output = tmp_path / "firm_year.csv"
            pd.DataFrame(
                [
                    {"text_unit_id": "id1", "stock_code": "1", "company_name": "A", "year": "2024", "source_file": "a.txt", "chapter_title": "经营情况", "unit_order": 1, "text": "x"},
                    {"text_unit_id": "id2", "stock_code": "2", "company_name": "B", "year": "2024", "source_file": "b.txt", "chapter_title": "经营情况", "unit_order": 1, "text": "y"},
                ]
            ).to_csv(text_units, index=False, encoding="utf-8-sig")
            pd.DataFrame(
                [
                    {
                        "text_unit_id": "id1",
                        "stock_code": "000001",
                        "company_name": "A",
                        "year": "2024",
                        "entity": "ISO 9001",
                        "type": "TYPE_A",
                        "status": "ADOPTED",
                        "evidence": "通过 ISO 9001",
                        "国际标准": "ISO 9001",
                        "采标情况": "TYPE_A",
                        "output": 1,
                    }
                ]
            ).to_csv(mapped, index=False, encoding="utf-8-sig")

            final = run_aggregate_main(text_units, mapped, output)

        self.assertEqual(final.loc[final["stock_code"] == "000001", "InternationalStandardDummy"].iloc[0], 1)
        self.assertEqual(final.loc[final["stock_code"] == "000002", "InternationalStandardDummy"].iloc[0], 0)
        self.assertEqual(final.loc[final["stock_code"] == "000001", "AdoptedEntityCount"].iloc[0], 1)
        self.assertNotIn("InternationalStandardShare", final.columns)
        self.assertNotIn("EntityCount", final.columns)

    def test_numbered_measurement_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            main = default_main_regression_paths(root, "2024")
            smoke = default_main_regression_paths(root, "2024", smoke=True)
            keyword = default_robustness_paths(root, "robustness_keyword", "2024")

        self.assertEqual(main.text_units_path, root / "data" / "measurement" / "main_regression" / "stage" / "01_text_units_2024.csv")
        self.assertEqual(main.final_output_path, root / "data" / "measurement" / "main_regression" / "final" / "07_main_regression_firm_year_2024.csv")
        self.assertEqual(smoke.base_dir, root / "data" / "measurement_smoke" / "main_regression")
        self.assertEqual(keyword.units_path, root / "data" / "measurement" / "robustness_keyword" / "stage" / "01_keyword_units_2024.csv")
        self.assertEqual(keyword.final_output_path, root / "data" / "measurement" / "robustness_keyword" / "final" / "04_robustness_keyword_firm_year_2024.csv")

    def test_prepare_robustness_units_are_method_specific(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            text_units = tmp_path / "01_text_units_2024.csv"
            keyword = tmp_path / "02_keyword_features_2024.csv"
            stage1 = tmp_path / "03_stage1_llm_relevance_2024.csv"
            rows = [
                {"text_unit_id": "id1", "stock_code": "000001", "company_name": "A", "year": "2024", "source_file": "a.txt", "chapter_title": "c", "unit_order": 1, "text": "keyword"},
                {"text_unit_id": "id2", "stock_code": "000001", "company_name": "A", "year": "2024", "source_file": "a.txt", "chapter_title": "c", "unit_order": 2, "text": "llm"},
                {"text_unit_id": "id3", "stock_code": "000001", "company_name": "A", "year": "2024", "source_file": "a.txt", "chapter_title": "c", "unit_order": 3, "text": "other"},
            ]
            pd.DataFrame(rows).to_csv(text_units, index=False, encoding="utf-8-sig")
            pd.DataFrame(
                [
                    {"text_unit_id": "id1", "keyword_candidate": True, "matched_terms": "ISO"},
                    {"text_unit_id": "id2", "keyword_candidate": False, "matched_terms": ""},
                    {"text_unit_id": "id3", "keyword_candidate": False, "matched_terms": ""},
                ]
            ).to_csv(keyword, index=False, encoding="utf-8-sig")
            pd.DataFrame(
                [
                    {"text_unit_id": "id1", "relevance": "unrelated", "confidence_score": 0.8, "reason": "", "stage1_status": "OK", "stage1_error": ""},
                    {"text_unit_id": "id2", "relevance": "related", "confidence_score": 0.9, "reason": "", "stage1_status": "OK", "stage1_error": ""},
                    {"text_unit_id": "id3", "relevance": "uncertain", "confidence_score": 0.5, "reason": "", "stage1_status": "OK", "stage1_error": ""},
                ]
            ).to_csv(stage1, index=False, encoding="utf-8-sig")

            keyword_units = run_prepare_keyword_units(text_units, keyword, tmp_path / "keyword_units.csv")
            llm_units = run_prepare_llm_units(text_units, stage1, tmp_path / "llm_units.csv")
            full_units = run_prepare_full_units(text_units, tmp_path / "full_units.csv")

        self.assertEqual(keyword_units["text_unit_id"].tolist(), ["id1"])
        self.assertEqual(llm_units["text_unit_id"].tolist(), ["id2", "id3"])
        self.assertEqual(full_units["text_unit_id"].tolist(), ["id1", "id2", "id3"])
        self.assertNotIn("keyword_candidate", keyword_units.columns)
        self.assertNotIn("relevance", llm_units.columns)

    def test_compare_dummy_consistency_outputs_expected_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = {
                "main": tmp_path / "main.csv",
                "keyword": tmp_path / "keyword.csv",
                "llm": tmp_path / "llm.csv",
                "full": tmp_path / "full.csv",
            }
            for key, path in paths.items():
                value = 0 if key == "keyword" else 1
                pd.DataFrame(
                    [
                        {
                            "stock_code": "000001",
                            "company_name": "A",
                            "year": "2024",
                            "InternationalStandardDummy": value,
                            "AdoptedEntityCount": value,
                        }
                    ]
                ).to_csv(path, index=False, encoding="utf-8-sig")

            collected_path = tmp_path / "01_collected.csv"
            consistency_path = tmp_path / "02_consistency.csv"
            collected = run_collect_firm_year_outputs(
                paths["main"],
                paths["keyword"],
                paths["llm"],
                paths["full"],
                collected_path,
            )
            consistency = run_compare_dummy_consistency(collected_path, consistency_path)

        self.assertIn("InternationalStandardDummy_full_llm", collected.columns)
        self.assertFalse(bool(consistency.loc[0, "main_eq_keyword"]))
        self.assertTrue(bool(consistency.loc[0, "main_eq_llm_only"]))
        self.assertTrue(bool(consistency.loc[0, "main_eq_full_llm"]))


if __name__ == "__main__":
    unittest.main()
