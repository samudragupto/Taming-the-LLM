from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    system: str
    user_template: str

    def render_user(self, **kwargs: str) -> str:
        return self.user_template.format(**kwargs)


SUMMARIZE = PromptTemplate(
    name="summarize",
    system="You are a concise summarizer. Produce a summary in plain English.",
    user_template="Summarize the following text in at most {max_sentences} sentences:\n\n{text}",
)

CLASSIFY_SENTIMENT = PromptTemplate(
    name="classify_sentiment",
    system=(
        "You are a sentiment classifier. "
        "Respond with exactly one word: positive, negative, or neutral."
    ),
    user_template="Classify the sentiment of the following text:\n\n{text}",
)

EXTRACT_ENTITIES = PromptTemplate(
    name="extract_entities",
    system=(
        "You are a named entity extractor. "
        "Return a JSON array of objects with keys 'entity' and 'type'. "
        "Types must be one of: PERSON, ORGANIZATION, LOCATION, DATE, OTHER."
    ),
    user_template="Extract all named entities from the following text:\n\n{text}",
)

ANSWER_QUESTION = PromptTemplate(
    name="answer_question",
    system=(
        "You are a helpful assistant. Answer the question based only on the provided context. "
        "If the context does not contain the answer, say 'I don't know'."
    ),
    user_template="Context:\n{context}\n\nQuestion:\n{question}",
)