"""Unit tests for the local examples and configuration errors in the tutorial."""

from __future__ import annotations

import contextlib
import io
import os
import unittest
from unittest.mock import Mock
from unittest.mock import patch

import langchain_latest_version_groq as tutorial
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage


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

    def test_demo_list_includes_profile_and_middleware_topics(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            tutorial.main(["--demo", "list"])

        self.assertIn("show-profile", output.getvalue())
        self.assertIn("dynamic-prompt", output.getvalue())
        self.assertIn("model-router", output.getvalue())
        self.assertIn("deep-agent-full", output.getvalue())

    def test_show_profile_agent_uses_trusted_runtime_context(self) -> None:
        fake_agent = Mock()
        fake_agent.invoke.return_value = {
            "messages": [Mock(content="User ID: 101, Role: admin")]
        }
        with (
            patch("langchain.agents.create_agent", return_value=fake_agent) as create_agent,
            patch.object(tutorial, "build_chat_model", return_value=object()),
        ):
            output = self.capture_output(tutorial.demo_show_profile)

        self.assertIs(create_agent.call_args.kwargs["context_schema"], tutorial.UserContext)
        self.assertIn(tutorial.show_profile, create_agent.call_args.kwargs["tools"])
        invocation = fake_agent.invoke.call_args
        self.assertEqual(invocation.kwargs["context"].user_id, "101")
        self.assertEqual(invocation.kwargs["context"].role, "admin")
        self.assertIn("User ID: 101, Role: admin", output)

    def test_show_profile_tool_receives_trusted_context_in_agent_runtime(self) -> None:
        class FakeToolCallingModel(FakeMessagesListChatModel):
            """Fake chat model that supports the agent's tool-binding step."""

            def bind_tools(self, tools, *, tool_choice=None, **kwargs):
                del tools, tool_choice, kwargs
                return self

        model = FakeToolCallingModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "show_profile",
                            "args": {},
                            "id": "profile-call-001",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="User ID: 101, Role: admin"),
            ]
        )
        agent = tutorial.create_profile_agent(model)

        result = agent.invoke(
            {"messages": [{"role": "user", "content": "Show my profile."}]},
            context=tutorial.UserContext(user_id="101", role="admin"),
        )

        tool_messages = [
            message for message in result["messages"] if isinstance(message, ToolMessage)
        ]
        self.assertEqual(len(tool_messages), 1)
        self.assertIn("User ID: 101", tool_messages[0].content)
        self.assertEqual(tool_messages[0].tool_call_id, "profile-call-001")
        self.assertEqual(result["messages"][-1].content, "User ID: 101, Role: admin")

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

    def test_structured_agent_demo_constructs_each_strategy(self) -> None:
        fake_agent = Mock()
        fake_agent.invoke.return_value = {"structured_response": "mock response"}
        with (
            patch("langchain.agents.create_agent", return_value=fake_agent) as create_agent,
            patch.object(tutorial, "build_chat_model", return_value=object()),
        ):
            self.capture_output(tutorial.demo_structured_agents)

        self.assertEqual(create_agent.call_count, 3)
        formats = [call.kwargs["response_format"] for call in create_agent.call_args_list]
        self.assertNotIsInstance(formats[0], (ProviderStrategy, ToolStrategy))
        self.assertIsInstance(formats[1], ProviderStrategy)
        self.assertIsInstance(formats[2], ToolStrategy)

    def test_manual_tool_call_returns_matching_tool_message(self) -> None:
        fake_model = Mock()
        fake_model.bind_tools.return_value.invoke.return_value = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "multiply",
                    "args": {"a": 12, "b": 8},
                    "id": "multiply-call-001",
                    "type": "tool_call",
                }
            ],
        )
        fake_model.invoke.return_value = AIMessage(content="12 times 8 is 96.")
        fake_agent = Mock()
        fake_agent.invoke.return_value = {
            "messages": [AIMessage(content="9 times 9 is 81.")]
        }
        with (
            patch.object(tutorial, "build_chat_model", return_value=fake_model),
            patch("langchain.agents.create_agent", return_value=fake_agent),
        ):
            output = self.capture_output(tutorial.demo_tools)

        conversation = fake_model.invoke.call_args.args[0]
        tool_response = next(
            message for message in conversation if isinstance(message, ToolMessage)
        )
        self.assertEqual(tool_response.content, "96")
        self.assertEqual(tool_response.tool_call_id, "multiply-call-001")
        self.assertIn("Manual tool-call round-trip answer", output)

    def test_prebuilt_middleware_demo_configures_notebook_examples(self) -> None:
        output = self.capture_output(tutorial.demo_prebuilt_middleware)

        self.assertIn("human_approval: HumanInTheLoopMiddleware", output)
        self.assertIn("pii_redaction: PIIMiddleware", output)
        self.assertIn("context_editing: ContextEditingMiddleware", output)

    def test_dynamic_prompt_demo_runs_with_fake_chat_model(self) -> None:
        class FakeToolCapableModel(FakeMessagesListChatModel):
            """Fake model that supports create_agent's bind-tools operation."""

            def bind_tools(self, tools, *, tool_choice=None, **kwargs):
                del tools, tool_choice, kwargs
                return self

        model = FakeToolCapableModel(
            responses=[
                AIMessage(content="Simple explanation."),
                AIMessage(content="Technical explanation."),
            ]
        )
        with patch.object(tutorial, "build_chat_model", return_value=model):
            output = self.capture_output(tutorial.demo_dynamic_prompt)

        self.assertIn("beginner answer: Simple explanation.", output)
        self.assertIn("expert answer: Technical explanation.", output)


if __name__ == "__main__":
    unittest.main()
