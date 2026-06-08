from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph_agent_toolkit.core.observability.factory import ObservabilityFactory
from langgraph_agent_toolkit.core.observability.types import ChatMessageDict, ObservabilityBackend
from langgraph_agent_toolkit.core.prompts.chat_prompt_template import (
    ObservabilityChatPromptTemplate,
    _convert_template_format,
)


class TestTemplateFormatConversion:
    """Tests for the _convert_template_format function."""

    @pytest.mark.parametrize(
        "template, target_format, expected",
        [
            # f-string -> jinja2
            ("Hello, {name}! Your age is {age}.", "jinja2", "Hello, {{ name }}! Your age is {{ age }}."),
            # jinja2 -> f-string
            ("Hello, {{ name }}! Your age is {{ age }}.", "f-string", "Hello, {name}! Your age is {age}."),
            # no variables -> unchanged in either direction
            ("Hello, no variables here!", "jinja2", "Hello, no variables here!"),
            ("Hello, no variables here!", "f-string", "Hello, no variables here!"),
            # already in the target format -> unchanged
            ("Hello, {{ name }}!", "jinja2", "Hello, {{ name }}!"),
            ("Hello, {name}!", "f-string", "Hello, {name}!"),
            # extra spaces inside jinja braces are normalized when converting to f-string
            ("Hello, {{  name  }}! Your age is {{age}}.", "f-string", "Hello, {name}! Your age is {age}."),
            # non-string inputs pass through untouched
            (None, "jinja2", None),
            (123, "f-string", 123),
        ],
    )
    def test_convert_template_format(self, template, target_format, expected):
        """_convert_template_format converts between f-string and jinja2, leaving non-templates untouched."""
        assert _convert_template_format(template, target_format) == expected


class TestObservabilityChatPromptTemplate:
    """Tests for the ObservabilityChatPromptTemplate class."""

    def setup_method(self):
        self.os_platform = ObservabilityFactory.create(
            ObservabilityBackend.EMPTY, remote_first=False
        )  # Create sample prompts
        basic_chat_messages: list[ChatMessageDict] = [
            {"role": "system", "content": "You are an AI assistant specialized in {{ domain }}."},
            {"role": "human", "content": "I need help with {{ question }} related to {{ topic }}."},
        ]
        self.os_platform.push_prompt("basic-assistant", basic_chat_messages)

        agent_chat_messages: list[ChatMessageDict] = [
            {"role": "system", "content": "You are an AI agent that can use tools. Available tools: {{ tools }}"},
            {"role": "human", "content": "I want to {{ task }} with the following context: {{ context }}"},
        ]
        self.os_platform.push_prompt("tool-using-agent", agent_chat_messages)

    def teardown_method(self):
        """Teardown for tests."""
        pass

    def test_init_from_observability_platform(self):
        """Test initialization from observability platform."""
        # Test with load_at_runtime=False (load during initialization)
        template = ObservabilityChatPromptTemplate.from_observability_platform(
            prompt_name="basic-assistant",
            observability_platform=self.os_platform,
            load_at_runtime=False,
            template_format="jinja2",
            input_variables=["domain", "question", "topic"],
        )

        assert template.prompt_name == "basic-assistant"
        assert template.observability_platform == self.os_platform
        assert template.load_at_runtime is False
        assert template.template_format == "jinja2"
        assert set(template.input_variables) == {"domain", "question", "topic"}

        # Verify messages were loaded at initialization time
        assert len(template.messages) == 2
        assert "specialized in {{ domain }}" in template.messages[0].prompt.template
        assert "{{ question }} related to {{ topic }}" in template.messages[1].prompt.template

        # Test with load_at_runtime=True
        runtime_template = ObservabilityChatPromptTemplate.from_observability_platform(
            prompt_name="tool-using-agent",
            observability_platform=self.os_platform,
            load_at_runtime=True,
            template_format="jinja2",
            input_variables=["tools", "task", "context"],
        )

        assert runtime_template.prompt_name == "tool-using-agent"
        assert runtime_template.load_at_runtime is True

        # Messages should be empty until invoked
        assert len(runtime_template.messages) == 0

    def test_init_from_observability_backend(self):
        """Test initialization from observability backend."""
        with patch(
            "langgraph_agent_toolkit.core.prompts.chat_prompt_template.ObservabilityFactory.create"
        ) as mock_create:
            mock_create.return_value = self.os_platform

            template = ObservabilityChatPromptTemplate.from_observability_backend(
                prompt_name="basic-assistant",
                observability_backend=ObservabilityBackend.EMPTY,
                load_at_runtime=True,
                template_format="jinja2",
                input_variables=["domain", "question", "topic"],
            )

            assert template.prompt_name == "basic-assistant"
            assert template.observability_backend == ObservabilityBackend.EMPTY
            assert template.load_at_runtime is True
            # The factory's result must actually be wired into the template, not just constructed.
            assert template.observability_platform is self.os_platform

            # Verify factory was called with correct backend
            mock_create.assert_called_once_with(ObservabilityBackend.EMPTY)

    def test_direct_initialization(self):
        """Test direct initialization."""
        template = ObservabilityChatPromptTemplate(
            prompt_name="tool-using-agent",
            observability_backend=ObservabilityBackend.EMPTY,
            load_at_runtime=True,
            template_format="jinja2",
            cache_ttl_seconds=120,
            input_variables=["tools", "task", "context"],
        )

        assert template.prompt_name == "tool-using-agent"
        assert template.observability_backend == ObservabilityBackend.EMPTY
        assert template.load_at_runtime is True
        assert template.template_format == "jinja2"
        assert template.cache_ttl_seconds == 120
        assert set(template.input_variables) == {"tools", "task", "context"}

    def test_invoke_init_time_loading(self):
        """Test invoking a template with initialization-time loading."""
        template = ObservabilityChatPromptTemplate.from_observability_platform(
            prompt_name="basic-assistant",
            observability_platform=self.os_platform,
            load_at_runtime=False,
            template_format="jinja2",
            input_variables=["domain", "question", "topic"],
        )

        result = template.invoke(
            input=dict(
                domain="machine learning",
                question="hyperparameter tuning",
                topic="neural networks",
            )
        )

        messages = result.to_messages()
        assert len(messages) == 2
        assert "specialized in machine learning" in messages[0].content
        assert "hyperparameter tuning related to neural networks" in messages[1].content

    def test_invoke_runtime_loading(self):
        """Test invoking a template with runtime loading."""
        template = ObservabilityChatPromptTemplate.from_observability_platform(
            prompt_name="tool-using-agent",
            observability_platform=self.os_platform,
            load_at_runtime=True,
            template_format="jinja2",
            input_variables=["tools", "task", "context"],
        )

        # Before first invocation, messages should be empty
        assert len(template.messages) == 0

        # First invocation should load the template
        result = template.invoke(
            input=dict(
                tools="search, calculator, weather",
                task="analyze market trends",
                context="technology sector in 2023",
            )
        )

        messages = result.to_messages()
        assert len(messages) == 2
        assert "Available tools: search, calculator, weather" in messages[0].content
        assert "analyze market trends" in messages[1].content
        assert "technology sector in 2023" in messages[1].content

        # After invocation, the template should be loaded
        assert len(template.messages) > 0

    @patch("langgraph_agent_toolkit.core.prompts.chat_prompt_template.logger")
    def test_error_handling(self, mock_logger):
        """Test error handling during prompt loading."""
        # Create a template with a nonexistent prompt
        template = ObservabilityChatPromptTemplate.from_observability_platform(
            prompt_name="nonexistent-prompt",
            observability_platform=self.os_platform,
            load_at_runtime=True,
            template_format="jinja2",
            input_variables=[],
        )

        # Should raise an error when trying to invoke
        with pytest.raises(ValueError, match="Failed to load prompt and no fallback available"):
            template.invoke(input={})

        # Logger should record the error
        mock_logger.error.assert_called_once()

        # Now create a template with fallback messages
        template_with_fallback = ObservabilityChatPromptTemplate(
            messages=[SystemMessage(content="Fallback message")],
            prompt_name="nonexistent-prompt",
            observability_platform=self.os_platform,
            load_at_runtime=True,
        )

        # Should use fallback messages without error
        result = template_with_fallback.invoke(input={})
        messages = result.to_messages()
        assert len(messages) == 1
        assert messages[0].content == "Fallback message"

    def test_switching_platform_reloads_prompt(self):
        """Assigning a new observability_platform resets cached state so the next invoke reloads from it.

        Drives only the public `observability_platform` setter (no poking of private fields).
        """
        template = ObservabilityChatPromptTemplate.from_observability_platform(
            prompt_name="basic-assistant",
            observability_platform=self.os_platform,
            load_at_runtime=True,
            template_format="jinja2",
            input_variables=["domain", "question", "topic"],
        )

        inputs = dict(domain="physics", question="relativity", topic="spacetime")
        first = template.invoke(input=inputs).to_messages()[0].content
        assert "MODIFIED" not in first

        new_platform = ObservabilityFactory.create(ObservabilityBackend.EMPTY)
        new_platform.push_prompt(
            "basic-assistant",
            [
                {"role": "system", "content": "You are a MODIFIED assistant specializing in {{ domain }}."},
                {"role": "human", "content": "I need help with {{ question }} related to {{ topic }}."},
            ],
        )

        # Public setter resets the cache; the next invoke must pull from the new platform.
        template.observability_platform = new_platform
        second = template.invoke(input=inputs).to_messages()[0].content
        assert "MODIFIED" in second

    def test_combining_with_standard_template(self):
        """Test combining with a standard ChatPromptTemplate."""
        observability_template = ObservabilityChatPromptTemplate.from_observability_platform(
            prompt_name="basic-assistant",
            observability_platform=self.os_platform,
            load_at_runtime=False,
            template_format="jinja2",
            input_variables=["domain", "question", "topic"],
        )

        standard_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content="Additional context: This is a follow-up question."),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessage(content="Can you elaborate on {{ specific_point }}?"),
            ],
            template_format="jinja2",
        )

        # Combine templates
        combined_template = observability_template + standard_template

        # When we check the length, we need to account for the chat_history expansion
        # The chat_history has 2 messages, so we'll get total of 5 messages when formatted
        result = combined_template.invoke(
            input=dict(
                domain="programming",
                question="debugging techniques",
                topic="Python",
                chat_history=[
                    HumanMessage(content="What are common debugging approaches?"),
                    AIMessage(content="Let me explain some debugging techniques."),
                ],
                specific_point="using breakpoints effectively",
            )
        )

        messages = result.to_messages()

        # Adjust the test to match the actual result
        assert len(messages) == 6
        assert "specialized in programming" in messages[0].content
        assert "debugging techniques related to Python" in messages[1].content
        assert "Additional context" in messages[2].content
        assert "common debugging approaches" in messages[3].content
        assert "explain some debugging techniques" in messages[4].content
        assert "using breakpoints effectively" in messages[5].content

    def test_partial_variables(self):
        """Test using partial variables."""
        template = ObservabilityChatPromptTemplate.from_observability_platform(
            prompt_name="basic-assistant",
            observability_platform=self.os_platform,
            load_at_runtime=False,
            template_format="jinja2",
            partial_variables={"domain": "fixed domain"},
            input_variables=["question", "topic"],
        )

        result = template.invoke(
            input=dict(
                question="partial variable test",
                topic="templates",
            )
        )

        messages = result.to_messages()
        assert "specialized in fixed domain" in messages[0].content
        assert "partial variable test related to templates" in messages[1].content

    def test_caching_reloads_only_after_ttl(self):
        """A runtime template re-pulls the prompt only after its cache TTL expires.

        Uses the real EmptyObservability backend plus a controllable clock, and asserts
        the observable effect (the rendered content changes after a reload) rather than
        mocked pull_prompt call counts.
        """
        platform = ObservabilityFactory.create(ObservabilityBackend.EMPTY)
        platform.push_prompt("cached", [{"role": "system", "content": "v1 {{ x }}"}])

        template = ObservabilityChatPromptTemplate.from_observability_platform(
            prompt_name="cached",
            observability_platform=platform,
            load_at_runtime=True,
            template_format="jinja2",
            cache_ttl_seconds=60,
            input_variables=["x"],
        )

        clock = {"now": 1000.0}
        with patch("time.time", lambda: clock["now"]):
            # First invoke loads v1.
            assert "v1 a" in template.invoke(input={"x": "a"}).to_messages()[0].content

            # Update the stored prompt to v2 behind the template's back.
            platform.push_prompt("cached", [{"role": "system", "content": "v2 {{ x }}"}])

            # Within the TTL window -> still serves cached v1 (no reload).
            clock["now"] += 30
            assert "v1 b" in template.invoke(input={"x": "b"}).to_messages()[0].content

            # Past the TTL -> reloads and serves v2.
            clock["now"] += 40  # +70s total, exceeds the 60s TTL
            assert "v2 c" in template.invoke(input={"x": "c"}).to_messages()[0].content

    @pytest.mark.asyncio
    async def test_async_invoke(self):
        """Test asynchronous invocation."""
        template = ObservabilityChatPromptTemplate.from_observability_platform(
            prompt_name="basic-assistant",
            observability_platform=self.os_platform,
            load_at_runtime=False,
            template_format="jinja2",
            input_variables=["domain", "question", "topic"],
        )

        result = await template.ainvoke(
            input=dict(
                domain="async programming",
                question="event loops",
                topic="concurrency",
            )
        )

        messages = result.to_messages()
        assert len(messages) == 2
        assert "specialized in async programming" in messages[0].content
        assert "event loops related to concurrency" in messages[1].content

    def test_format_conversion_in_templates(self):
        """Test format conversion within templates."""
        # Create a template with f-string format
        # Note: Format conversion happens during invoke, not during template loading
        f_string_messages = [
            {"role": "system", "content": "You are an assistant for {domain}."},
            {"role": "human", "content": "Help with {topic}"},
        ]

        self.os_platform.push_prompt("f-string-template", f_string_messages)

        # Load with f-string format (matching the content format)
        template = ObservabilityChatPromptTemplate.from_observability_platform(
            prompt_name="f-string-template",
            observability_platform=self.os_platform,
            load_at_runtime=False,
            template_format="f-string",
            input_variables=["domain", "topic"],
        )

        # The template should be loaded with f-string format
        assert "{domain}" in template.messages[0].prompt.template
        assert "{topic}" in template.messages[1].prompt.template

        # Invoke to verify it works with f-string format
        result = template.invoke(
            input=dict(
                domain="format conversion",
                topic="template systems",
            )
        )

        messages = result.to_messages()
        assert "assistant for format conversion" in messages[0].content
        assert "Help with template systems" in messages[1].content

    def test_add_operator_with_different_types(self):
        """Test the __add__ operator with different types of inputs."""
        # Create base template
        base_template = ObservabilityChatPromptTemplate.from_observability_platform(
            prompt_name="basic-assistant",
            observability_platform=self.os_platform,
            load_at_runtime=False,
            template_format="jinja2",
            input_variables=["domain", "question", "topic"],
        )

        # Test adding a BaseMessage
        message_result = base_template + HumanMessage(content="Additional question: {{ follow_up }}")
        assert len(message_result.messages) == 3
        # Fixed: access prompt.template instead of content for message templates
        if hasattr(message_result.messages[2], "prompt"):
            assert "Additional question:" in message_result.messages[2].prompt.template
        else:
            assert "Additional question:" in message_result.messages[2].content

        # Test adding a BaseMessagePromptTemplate
        from langchain_core.prompts.chat import HumanMessagePromptTemplate

        template_result = base_template + HumanMessagePromptTemplate.from_template(
            "Template question: {{ template_var }}", template_format="jinja2"
        )
        assert len(template_result.messages) == 3
        assert "Template question:" in template_result.messages[2].prompt.template

        # Test adding a list of messages
        list_result = base_template + [
            SystemMessage(content="System note: {{ note }}"),
            HumanMessage(content="Human followup: {{ followup }}"),
        ]
        assert len(list_result.messages) == 4
        # Fixed: Properly check content in messages
        if hasattr(list_result.messages[2], "prompt"):
            assert "System note:" in list_result.messages[2].prompt.template
            assert "Human followup:" in list_result.messages[3].prompt.template
        else:
            assert "System note:" in list_result.messages[2].content
            assert "Human followup:" in list_result.messages[3].content

        # Test adding a string (should create HumanMessagePromptTemplate)
        # FIX: Create a proper ChatPromptTemplate with the right template format first
        from langchain_core.prompts import ChatPromptTemplate

        string_template = ChatPromptTemplate.from_template(
            "Simple string message: {{ simple_var }}", template_format="jinja2"
        )
        string_result = base_template + string_template

        assert len(string_result.messages) == 3
        assert "Simple string message:" in string_result.messages[2].prompt.template

        # Fix: Test that the template actually works with the variable
        result = string_result.invoke(
            {"domain": "test domain", "question": "test question", "topic": "test topic", "simple_var": "test variable"}
        )
        messages = result.to_messages()
        assert len(messages) == 3
        assert "Simple string message: test variable" in messages[2].content

        # Test invalid addition
        with pytest.raises(NotImplementedError):
            base_template + 123

    def test_message_placeholder_handling(self):
        """Test handling of MessagesPlaceholder in different scenarios."""
        # Push this template to the observability platform so we can load it in ObservabilityChatPromptTemplate
        chat_messages = [
            {"role": "system", "content": "System prompt with {{ system_var }}"},
            {"role": "messages_placeholder", "content": "history"},
            {"role": "human", "content": "Human prompt with {{ human_var }}"},
        ]
        self.os_platform.push_prompt("message-placeholder-test", chat_messages)

        # Load it through ObservabilityChatPromptTemplate
        template = ObservabilityChatPromptTemplate.from_observability_platform(
            prompt_name="message-placeholder-test",
            observability_platform=self.os_platform,
            load_at_runtime=False,
            template_format="jinja2",
            input_variables=["system_var", "human_var", "history"],
        )

        # Test with empty history
        result1 = template.invoke(input=dict(system_var="system value", human_var="human value", history=[]))

        messages1 = result1.to_messages()
        assert len(messages1) == 2  # No history messages (system + human)
        assert "System prompt with system value" in messages1[0].content
        assert "Human prompt with human value" in messages1[1].content

        # Test with history containing messages
        result2 = template.invoke(
            input=dict(
                system_var="system value",
                human_var="human value",
                history=[
                    HumanMessage(content="Previous question"),
                    AIMessage(content="Previous answer"),
                    HumanMessage(content="Follow-up question"),
                    AIMessage(content="Follow-up answer"),
                ],
            )
        )

        messages2 = result2.to_messages()
        assert len(messages2) == 6  # System + 4 history + Human
        assert "System prompt with system value" in messages2[0].content
        assert "Previous question" in messages2[1].content
        assert "Follow-up answer" in messages2[4].content
        assert "Human prompt with human value" in messages2[5].content

    def test_process_list_prompt_formats(self):
        """A dict-role list loads into the correct message-template types, in order."""
        from langchain_core.prompts.chat import (
            AIMessagePromptTemplate,
            HumanMessagePromptTemplate,
            SystemMessagePromptTemplate,
        )

        dict_prompts = [
            {"role": "system", "content": "System message with {{ var1 }}"},
            {"role": "human", "content": "Human message with {{ var2 }}"},
            {"role": "assistant", "content": "Assistant message with {{ var3 }}"},
        ]
        self.os_platform.push_prompt("dict-format-simple", dict_prompts)

        template = ObservabilityChatPromptTemplate.from_observability_platform(
            prompt_name="dict-format-simple",
            observability_platform=self.os_platform,
            load_at_runtime=False,
            template_format="jinja2",
            input_variables=["var1", "var2", "var3"],
        )

        # (MessagesPlaceholder handling within a dict list is covered by test_message_placeholder_handling.)
        assert len(template.messages) == 3
        assert isinstance(template.messages[0], SystemMessagePromptTemplate)
        assert isinstance(template.messages[1], HumanMessagePromptTemplate)
        assert isinstance(template.messages[2], AIMessagePromptTemplate)


class TestChatPromptValueValidation:
    """Tests to ensure ChatPromptValue always receives BaseMessage instances.

    This test class verifies the fix for the validation error:
    'Input should be a valid dictionary or instance of BaseMessage'
    which occurred when SystemMessagePromptTemplate was passed to ChatPromptValue
    instead of properly formatted BaseMessage.
    """

    def setup_method(self):
        """Setup for each test method."""  # noqa: D401
        self.os_platform = ObservabilityFactory.create(ObservabilityBackend.EMPTY, remote_first=False)
        # Create a test prompt
        test_messages: list[ChatMessageDict] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "placeholder", "content": "messages"},
        ]
        self.os_platform.push_prompt("test-prompt", test_messages)

    def teardown_method(self):
        """Teardown for tests."""
        pass

    @pytest.mark.parametrize("use_async", [False, True])
    async def test_invoke_returns_only_base_messages(self, use_async):
        """invoke()/ainvoke() must place only BaseMessage instances in the ChatPromptValue."""
        from langchain_core.messages import BaseMessage
        from langchain_core.prompts.chat import SystemMessagePromptTemplate

        template = ObservabilityChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template("You are a helpful assistant", template_format="jinja2"),
                MessagesPlaceholder(variable_name="messages"),
            ],
            input_variables=["messages"],
            template_format="jinja2",
        )

        payload = {"messages": [HumanMessage(content="Hello!")]}
        result = await template.ainvoke(payload) if use_async else template.invoke(payload)

        for msg in result.messages:
            assert isinstance(msg, BaseMessage), f"Expected BaseMessage, got {type(msg)}"
