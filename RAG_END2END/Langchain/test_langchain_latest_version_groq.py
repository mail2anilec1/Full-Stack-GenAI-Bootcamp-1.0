"""Unit tests for the local examples and configuration errors in the tutorial."""

from __future__ import annotations

import contextlib
import io
import os
import unittest
from unittest.mock import Mock
from unittest.mock import patch

import langchain_latest_version_groq as tutorial
from langchain.agents.structured_output import ToolStrategy


class TutorialDemoTests(unittest.TestCase):
    """Verify demos that do not require a live Groq API request."""

    def capture_output(self, function) -> str:
        """Run a demo and return its standard output."""
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            function()
        return output.getvalue()

    def test_document_demo_prints_content_and_metadata(self) -> None:
        output = self.capture_output(tutorial.demo_document)

        self.assertIn("LangChain provides composable building blocks.", output)
        self.assertIn("'source': 'course_notes'", output)
        self.assertIn("doc-001", output)

    def test_messages_demo_shows_trimmed_history(self) -> None:
        output = self.capture_output(tutorial.demo_messages)

        self.assertIn("Recent conversation", output)
        self.assertIn("Teach me RAG with an example.", output)
        self.assertNotIn("HumanMessage: Hello", output)

    def test_prompts_demo_formats_prompt_variables(self) -> None:
        output = self.capture_output(tutorial.demo_prompts)

        self.assertIn("Explain embeddings to a beginner learner.", output)
        self.assertIn("Explain at expert level.", output)
        self.assertIn("Teach retrievers.", output)

    def test_main_dispatches_selected_local_demo(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            tutorial.main(["--demo", "prompts"])

        self.assertIn("PROMPTS", output.getvalue())
        self.assertIn("Explain embeddings to a beginner learner.", output.getvalue())

    def test_chat_model_requires_groq_api_key(self) -> None:
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "GROQ_API_KEY is not set"):
                tutorial.build_chat_model()

    def test_deep_agent_reports_missing_optional_dependency(self) -> None:
        missing_dependency = ModuleNotFoundError(
            "No module named 'deepagents'",
            name="deepagents",
        )
        with patch.object(tutorial, "import_module", side_effect=missing_dependency):
            with self.assertRaisesRegex(RuntimeError, "pip install -U deepagents"):
                tutorial.demo_deep_agent()

    def test_structured_agent_uses_tool_strategy_with_groq(self) -> None:
        fake_agent = Mock()
        fake_agent.invoke.return_value = {"structured_response": "mock response"}
        with (
            patch("langchain.agents.create_agent", return_value=fake_agent) as create_agent,
            patch.object(tutorial, "build_chat_model", return_value=object()),
        ):
            output = self.capture_output(tutorial.demo_agents)

        response_format = create_agent.call_args.kwargs["response_format"]
        self.assertIsInstance(response_format, ToolStrategy)
        self.assertIn("mock response", output)


if __name__ == "__main__":
    unittest.main()
