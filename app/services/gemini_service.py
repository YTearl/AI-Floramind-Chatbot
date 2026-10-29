import logging
from typing import Any, Dict, List, Optional
from app.config import settings
from app.models.schemas import ChatMessage

logger = logging.getLogger(__name__)

FLORAMIND_SYSTEM_PROMPT = """You are Floramind, a warm, friendly, and knowledgeable gardening mentor created to guide beginners on their growing journey.

Core Principles & Personality:
1. Beginner-Friendly & Clear: Use simple, everyday English. Explain gardening concepts step-by-step. If a technical gardening term is needed (like 'hardening off' or 'loam'), explain it immediately in plain language.
2. Practical & Actionable: Give practical, actionable advice on seeds, soil, watering, sunlight, fertilizers, compost, vegetables, herbs, flowers, pests, and plant diseases.
3. Clarifying Questions: Ask friendly clarifying questions when the specific plant species, local climate, indoor/outdoor setup, or container size would change the advice.
4. Tropical Gardening: Provide specialized tropical gardening advice (addressing high humidity, heavy monsoon rains, soil nutrient leaching, raised drainage beds, and tropical plants like kangkong, bougainvillea, hibiscus, lemongrass, calamansi) when relevant or supported by knowledge base references.
5. Strict Grounding & Source Integrity:
   - You are provided with retrieved gardening reference documents.
   - Use these retrieved references to answer the user's question accurately.
   - Clearly distinguish facts retrieved directly from the knowledge base from general gardening guidance.
   - NEVER invent sources, fabricate citations, or claim that unsupported information came from the knowledge base.
   - If the retrieved references are insufficient, do not contain the answer, or retrieval returned no close matches, explicitly state that your gardening knowledge base does not have specific indexed documents on that topic. Then, provide safe, friendly general gardening advice while being transparent about the limitation.
6. Safety & Responsibility:
   - Provide safe, responsible advice regarding toxic plants (especially plants toxic to dogs, cats, or small children like lilies, oleander, or dieffenbachia).
   - Prioritize organic, cultural, and mechanical pest controls first (neem oil, soapy water, beneficial insects, proper airflow).
   - If pesticides or chemical treatments are discussed, emphasize wearing protective gear, strictly adhering to label directions, and avoiding spraying near pollinators or before rain.
"""


class GeminiService:
    """Dedicated service for Google Gemini API integration using the official google-genai SDK.
    
    Supports model configuration via environment variable, prefers the Interactions API
    when available, and provides robust fallback and error handling.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        mock_mode: bool = False,
    ):
        self.api_key = api_key if api_key is not None else settings.GEMINI_API_KEY
        self.model_name = model_name if model_name is not None else settings.GEMINI_MODEL
        self.mock_mode = mock_mode
        self._client = None

        if not self.mock_mode and self.is_configured:
            self._init_client()

    @property
    def is_configured(self) -> bool:
        """Check if a non-placeholder Gemini API key is configured."""
        return bool(
            self.api_key
            and self.api_key != "your_gemini_api_key"
            and len(self.api_key.strip()) > 5
        )

    def _init_client(self) -> None:
        """Initialize the official Google GenAI SDK client."""
        try:
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
            logger.info(
                f"Google GenAI client initialized with model '{self.model_name}'."
            )
        except Exception as e:
            logger.error(f"Failed to initialize Google GenAI client: {e}")
            self._client = None

    def build_prompt(
        self,
        query: str,
        context_docs: List[Dict[str, Any]],
        is_relevant: bool = True,
    ) -> str:
        """Construct the prompt incorporating retrieved knowledge and instructions."""
        context_text_blocks = []
        if context_docs:
            for idx, doc in enumerate(context_docs, 1):
                doc_title = doc.get("title", f"Document #{idx}")
                doc_category = doc.get("category", "Gardening")
                doc_section = doc.get("section", "")
                doc_text = doc.get("text", "")
                header = f"--- Reference [{idx}]: {doc_title} ({doc_category}"
                if doc_section:
                    header += f" - {doc_section}"
                header += ") ---"
                context_text_blocks.append(f"{header}\n{doc_text}\n")
            joined_context = "\n".join(context_text_blocks)
        else:
            joined_context = "No relevant gardening documents were retrieved for this query."

        prompt = f"""KNOWLEDGE BASE REFERENCES:
{joined_context}

RELEVANCE STATUS: {"HIGHLY RELEVANT" if is_relevant and context_docs else "INSUFFICIENT OR LOW RELEVANCE"}

USER'S GARDENING QUESTION:
{query}

INSTRUCTIONS FOR FLORAMIND:
1. Answer the user's question warmly and clearly as Floramind.
2. If RELEVANCE STATUS is 'HIGHLY RELEVANT', base your answer primarily on the Knowledge Base References. Mention key principles and cite concepts directly.
3. If RELEVANCE STATUS is 'INSUFFICIENT OR LOW RELEVANCE', explicitly inform the user that the knowledge base does not currently contain detailed documents for this specific inquiry, but provide safe and helpful general gardening guidance.
4. Include any necessary safety cautions (e.g. toxicity to pets/children, safe pesticide usage).
5. Conclude with a helpful next step or offer to clarify plant variety/growing conditions.
"""
        return prompt

    def generate_response(
        self,
        query: str,
        context_docs: List[Dict[str, Any]],
        history: Optional[List[ChatMessage]] = None,
        is_relevant: bool = True,
    ) -> Dict[str, Any]:
        """Generate conversational answer via Google Gemini API.
        
        Attempts the recommended Interactions API first; falls back to
        client.models.generate_content if needed.
        """
        # If running in mock mode or unconfigured, provide structured mock response
        if self.mock_mode or not self.is_configured or self._client is None:
            return self._generate_mock_response(query, context_docs, is_relevant)

        prompt = self.build_prompt(query, context_docs, is_relevant=is_relevant)

        # Attempt 1: Recommended Interactions API
        try:
            logger.info(f"Generating answer via Gemini Interactions API ({self.model_name})...")
            # Format input combining system instruction and prompt for unified session
            interaction_input = f"[System Context: {FLORAMIND_SYSTEM_PROMPT}]\n\n{prompt}"
            
            # Format conversation history if provided
            if history:
                history_text = "\n".join(
                    [f"{msg.role.capitalize()}: {msg.content}" for msg in history[-4:]]
                )
                interaction_input = (
                    f"[System Context: {FLORAMIND_SYSTEM_PROMPT}]\n\n"
                    f"Previous Conversation:\n{history_text}\n\n{prompt}"
                )

            interaction = self._client.interactions.create(
                model=self.model_name,
                input=interaction_input,
            )

            # Check output text attribute
            answer_text = None
            if hasattr(interaction, "output_text") and interaction.output_text:
                answer_text = interaction.output_text
            elif hasattr(interaction, "outputs") and interaction.outputs:
                # Some versions provide outputs array
                answer_text = str(interaction.outputs[0])
            elif hasattr(interaction, "text") and interaction.text:
                answer_text = interaction.text

            if answer_text:
                return {
                    "answer": answer_text.strip(),
                    "model_used": self.model_name,
                    "method": "interactions_api",
                }

            logger.info("Interactions API returned empty text; trying client.models.generate_content.")
        except Exception as inter_err:
            logger.info(
                f"Interactions API not available or raised error ({inter_err}); "
                f"switching to client.models.generate_content."
            )

        # Attempt 2: Standard client.models.generate_content
        try:
            from google.genai import types
            from google.genai.errors import APIError, ClientError

            # Build multi-turn content if history exists
            contents = []
            if history:
                for h_msg in history[-6:]:  # Keep recent context
                    role = "user" if h_msg.role == "user" else "model"
                    contents.append(types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=h_msg.content)]
                    ))
            
            # Add latest prompt
            contents.append(types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)]
            ))

            config = types.GenerateContentConfig(
                system_instruction=FLORAMIND_SYSTEM_PROMPT,
                temperature=0.6,
                max_output_tokens=1000,
            )

            response = self._client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )

            if response and response.text:
                return {
                    "answer": response.text.strip(),
                    "model_used": self.model_name,
                    "method": "models_generate_content",
                }
            else:
                raise ValueError("Gemini returned an empty response.")

        except Exception as e:
            logger.error(f"Gemini API error during generation: {e}")
            return self._handle_gemini_error(e, query, context_docs, is_relevant)

    def _handle_gemini_error(
        self,
        exc: Exception,
        query: str,
        context_docs: List[Dict[str, Any]],
        is_relevant: bool,
    ) -> Dict[str, Any]:
        """Translates API errors into helpful, user-friendly guidance."""
        err_msg = str(exc).lower()

        if "api_key" in err_msg or "401" in err_msg or "unauthenticated" in err_msg:
            friendly_msg = (
                "🌿 **Notice from Floramind:** The Google Gemini API key appears to be invalid or unauthenticated. "
                "Please configure a valid `GEMINI_API_KEY` in your `.env` file to enable live Gemini generation."
            )
        elif "quota" in err_msg or "429" in err_msg or "resource_exhausted" in err_msg:
            friendly_msg = (
                "🌿 **Notice from Floramind:** The Gemini API quota or rate limit has been temporarily exceeded. "
                "Please wait a moment before asking your next gardening question."
            )
        elif "not found" in err_msg or "404" in err_msg or "model" in err_msg:
            friendly_msg = (
                f"🌿 **Notice from Floramind:** The configured model `{self.model_name}` could not be reached. "
                "Check that the model name in `GEMINI_MODEL` is valid and supported by your API tier."
            )
        elif "connection" in err_msg or "timeout" in err_msg or "network" in err_msg:
            friendly_msg = (
                "🌿 **Notice from Floramind:** A network connection error occurred while reaching the Gemini API. "
                "Please check your internet connection."
            )
        else:
            friendly_msg = (
                f"🌿 **Notice from Floramind:** An unexpected error occurred while communicating with Gemini ({exc}). "
                "Here is guidance compiled directly from our gardening references:"
            )

        # Append reference summaries if available
        if context_docs and is_relevant:
            refs_summary = "\n\n**Retrieved Gardening References:**\n"
            for doc in context_docs[:3]:
                refs_summary += f"- **{doc.get('title')}**: {doc.get('text')[:180]}...\n"
            friendly_msg += refs_summary

        return {
            "answer": friendly_msg,
            "model_used": self.model_name,
            "method": "error_fallback",
            "error": str(exc),
        }

    def _generate_mock_response(
        self,
        query: str,
        context_docs: List[Dict[str, Any]],
        is_relevant: bool,
    ) -> Dict[str, Any]:
        """Simulate Floramind gardening mentor response for testing and local setups."""
        if not is_relevant or not context_docs:
            answer = (
                f"Hello there! I'm Floramind, your gardening mentor. 🌿\n\n"
                f"I looked into our indexed knowledge base, but I don't currently have detailed reference documents "
                f"specifically covering: *\"{query}\"*.\n\n"
                f"As general gardening guidance, always ensure your plants receive proper drainage, inspect "
                f"the top inch of soil for moisture before watering, and ensure adequate light for your species. "
                f"Could you share a bit more detail about the specific plant or your climate?"
            )
        else:
            top_doc = context_docs[0]
            answer = (
                f"Hello fellow gardener! I'm Floramind. 🌿 Let's explore your question: *\"{query}\"*.\n\n"
                f"Based on our gardening reference **{top_doc.get('title')}** ({top_doc.get('category')}):\n\n"
                f"{top_doc.get('text')}\n\n"
            )
            if len(context_docs) > 1:
                second_doc = context_docs[1]
                answer += (
                    f"**Helpful Tip from {second_doc.get('title')}:**\n"
                    f"{second_doc.get('text')}\n\n"
                )
            answer += (
                "**Step-by-Step Action:**\n"
                "1. Check the moisture and condition of your soil or plant.\n"
                "2. Apply these steps gently and observe how your plant responds over the next 3 to 5 days.\n\n"
                "Feel free to let me know your specific climate or plant species if you need tailored advice!"
            )

        return {
            "answer": answer,
            "model_used": f"{self.model_name} (local/mock)",
            "method": "mock_generator",
        }


_gemini_service_instance: Optional[GeminiService] = None

def get_gemini_service() -> GeminiService:
    """Singleton accessor for GeminiService."""
    global _gemini_service_instance
    if _gemini_service_instance is None:
        _gemini_service_instance = GeminiService()
    return _gemini_service_instance
