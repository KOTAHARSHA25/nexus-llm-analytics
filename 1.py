from backend.core.llm_client import LLMClient
llm = LLMClient()
print(llm.generate_primary("Say hello!"))
print(llm.generate_review("Review this code: print('hi')"))