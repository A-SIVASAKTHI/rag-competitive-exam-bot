# utils.py

import re


def split_multiple_questions(text):
    """
    Splits combined questions properly.
    Handles:
    - and
    - ?
    """
    questions = re.split(r'\?|\band\b', text, flags=re.IGNORECASE)
    return [q.strip() for q in questions if q.strip()]


def format_answer_clean(question, answer):
    """
    Returns neatly formatted structured answer.
    """

    return f"""
    <div class="qa-block">
        <div class="question-title">❓ {question.capitalize()}</div>
        <div class="answer-content">
            {answer.strip()}
        </div>
    </div>
    """
