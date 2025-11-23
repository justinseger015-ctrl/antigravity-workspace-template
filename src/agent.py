import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from google import genai

from src.config import settings
from src.memory import MemoryManager
from src.tools.example_tool import get_stock_price, web_search


class GeminiAgent:
    """
    A production-grade agent wrapper for Gemini 3.
    Implements the Think-Act-Reflect loop.
    """

    def __init__(self):
        self.settings = settings
        self.memory = MemoryManager()
        # Register tools here to make them available to the agent.
        # To add a new tool, import it above and append to this mapping.
        self.available_tools: Dict[str, Callable[..., Any]] = {
            "get_stock_price": get_stock_price,
            "web_search": web_search,
        }
        print(f"🤖 Initializing {self.settings.AGENT_NAME} with model {self.settings.GEMINI_MODEL_NAME}...")
        self.client = genai.Client(api_key=self.settings.GOOGLE_API_KEY)

    def _get_tool_descriptions(self) -> str:
        """
        Dynamically builds a list of available tools and their docstrings for prompt injection.
        """
        descriptions: List[str] = []
        for name, fn in self.available_tools.items():
            doc = (fn.__doc__ or "No description provided.").strip().replace("\n", " ")
            descriptions.append(f"- {name}: {doc}")
        return "\n".join(descriptions)

    def _format_context_messages(self, context_messages: List[Dict[str, Any]]) -> str:
        """
        Flattens structured context into a plain-text prompt block.
        """
        lines = [f"{msg.get('role', '').upper()}: {msg.get('content', '')}" for msg in context_messages]
        return "\n".join(lines)

    def _call_gemini(self, prompt: str) -> str:
        """Lightweight wrapper around the Gemini content generation call."""
        response_obj = self.client.models.generate_content(
            model=self.settings.GEMINI_MODEL_NAME,
            contents=prompt,
        )
        return response_obj.text.strip()

    def _extract_tool_call(self, response_text: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Parses a model response to detect a tool invocation request.

        Supports two patterns:
        1) JSON object: {"action": "tool_name", "args": {...}}
        2) Plain text line starting with 'Action: <tool_name>'
        """
        cleaned = response_text.strip()

        try:
            payload = json.loads(cleaned)
            if isinstance(payload, dict):
                action = payload.get("action") or payload.get("tool")
                args = payload.get("args") or payload.get("input") or {}
                if action:
                    return str(action), args if isinstance(args, dict) else {}
        except json.JSONDecodeError:
            pass

        for line in cleaned.splitlines():
            if line.lower().startswith("action:"):
                action = line.split(":", 1)[1].strip()
                if action:
                    return action, {}

        return None, {}

    def summarize_memory(self, old_messages: List[Dict[str, Any]], previous_summary: str) -> str:
        """
        Summarize older history into a concise buffer using Gemini.
        """
        history_block = "\n".join([f"- {m.get('role', 'unknown')}: {m.get('content', '')}" for m in old_messages])
        prompt = (
            "You are an expert conversation summarizer for an autonomous agent.\n"
            "Goals:\n"
            "1) Preserve decisions, intents, constraints, and outcomes.\n"
            "2) Omit small talk and low-signal chatter.\n"
            "3) Keep the summary under 120 words and in plain text.\n"
            "4) Maintain continuity so future turns understand what has already happened.\n\n"
            f"Previous summary:\n{previous_summary or '[none]'}\n\n"
            "Messages to summarize (oldest first):\n"
            f"{history_block}\n\n"
            "Return only the new merged summary."
        )

        response = self.client.models.generate_content(
            model=self.settings.GEMINI_MODEL_NAME,
            contents=[{"role": "user", "parts": [prompt]}],
        )
        return response.text.strip()

    def think(self, task: str) -> str:
        """
        Simulates the 'Deep Think' process of Gemini 3.
        """
        system_prompt = "You are a focused agent following the Artifact-First protocol. Stay concise and tactical."
        context_window = self.memory.get_context_window(
            system_prompt=system_prompt,
            max_messages=10,
            summarizer=self.summarize_memory,
        )

        print(f"\n🤔 <thought> Analyzing task: '{task}'")
        print(f"   - Loaded context messages: {len(context_window)}")
        print("   - Checking mission context...")
        print("   - Identifying necessary tools...")
        print("   - Formulating execution plan...")
        print("</thought>\n")

        time.sleep(1)
        return "Plan formulated."

    def act(self, task: str) -> str:
        """
        Executes the task using available tools and generates a real response.
        """
        # 1) Record user input
        self.memory.add_entry("user", task)

        # 2) Think
        self.think(task)

        # 3) Tool dispatch entry point
        print(f"[TOOLS] Executing tools for: {task}")
        tool_list = self._get_tool_descriptions()

        system_prompt = (
            "You are an expert AI agent following the Think-Act-Reflect loop.\n"
            "You have access to the following tools:\n"
            f"{tool_list}\n\n"
            "If you need a tool, respond ONLY with a JSON object using the schema:\n"
            '{"action": "<tool_name>", "args": {"param": "value"}}\n'
            "If no tool is needed, reply directly with the final answer."
        )

        try:
            context_messages = self.memory.get_context_window(
                system_prompt=system_prompt,
                max_messages=10,
                summarizer=self.summarize_memory,
            )
            formatted_context = self._format_context_messages(context_messages)
            initial_prompt = f"{formatted_context}\n\nCurrent Task: {task}"

            print("💬 Sending request to Gemini...")
            first_reply = self._call_gemini(initial_prompt)
            tool_name, tool_args = self._extract_tool_call(first_reply)

            final_response = first_reply

            if tool_name:
                tool_fn = self.available_tools.get(tool_name)
                if not tool_fn:
                    observation = f"Requested tool '{tool_name}' is not registered."
                else:
                    try:
                        observation = tool_fn(**tool_args)
                    except TypeError as exc:
                        observation = f"Error executing tool '{tool_name}': {exc}"
                    except Exception as exc:
                        observation = f"Unexpected error in tool '{tool_name}': {exc}"

                # Record intermediate reasoning and observation
                self.memory.add_entry("assistant", first_reply)
                self.memory.add_entry("tool", f"{tool_name} output: {observation}")

                # Refresh context to include tool feedback before final answer
                context_messages = self.memory.get_context_window(
                    system_prompt=system_prompt,
                    max_messages=10,
                    summarizer=self.summarize_memory,
                )
                formatted_context = self._format_context_messages(context_messages)
                follow_up_prompt = (
                    f"{formatted_context}\n\n"
                    f"Tool '{tool_name}' observation: {observation}\n"
                    "Use the observation above to craft the final answer for the user. "
                    "Do not request additional tool calls."
                )
                print(f"💬 Sending follow-up with observation from '{tool_name}'...")
                final_response = self._call_gemini(follow_up_prompt)

            self.memory.add_entry("assistant", final_response)
            return final_response

        except Exception as e:
            response = f"Error generating response: {str(e)}"
            print(f"❌ API Error: {e}")
            return response

    def reflect(self):
        """
        Review past actions to improve future performance.
        """
        history = self.memory.get_history()
        print(f"Reflecting on {len(history)} past interactions...")

    def run(self, task: str):
        """Main entry point for the agent."""
        print(f"🚀 Starting Task: {task}")
        result = self.act(task)
        print(f"📦 Result: {result}")
        self.reflect()


if __name__ == "__main__":
    agent = GeminiAgent()
    agent.run("Analyze the stock performance of GOOGL")
