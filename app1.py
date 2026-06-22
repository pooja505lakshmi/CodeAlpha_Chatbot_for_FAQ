import pandas as pd
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

faq = pd.read_csv("faq.csv")

def clean_text(text):

    text = text.lower()

    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    return text

faq["Cleaned"] = faq["Question"].apply(clean_text)

vectorizer = TfidfVectorizer()

faq_vectors = vectorizer.fit_transform(
    faq["Cleaned"]
)

def get_answer(user_question):

    cleaned_question = clean_text(
        user_question
    )

    user_vector = vectorizer.transform(
        [cleaned_question]
    )

    similarity_scores = cosine_similarity(
        user_vector,
        faq_vectors
    )

    best_match = similarity_scores.argmax()

    confidence = similarity_scores.max()

    if confidence < 0.20:

        return (
            "Sorry, I don't know the answer to that question.",
            confidence
        )

    answer = faq.iloc[best_match]["Answer"]

    return answer, confidence
if __name__ == "__main__":

    print("\n🎓 AI Career Mentor Chatbot")
    print("Type 'exit' to quit.\n")

    while True:

        question = input("You: ")

        if question.lower() == "exit":
            print("Bot: Goodbye! 👋")
            break

        answer, confidence = get_answer(question)

        print(f"Bot: {answer}")

        print(
            f"Confidence: {confidence:.2%}\n"
        )