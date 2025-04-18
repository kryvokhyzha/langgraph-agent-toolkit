from langfuse import Langfuse


# Initialize Langfuse client
langfuse = Langfuse()

# Get by label
# You can use as many labels as you'd like to identify different deployment targets
prompt = langfuse.get_prompt("react-assistant", label="production")
print(prompt.get_langchain_prompt())
print(prompt.prompt)
