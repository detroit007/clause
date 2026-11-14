import json
import time
from datetime import datetime
from pathlib import Path

from app.services.chat import RAGService


DATASET_PATH = "evals/golden_v2.jsonl"
RESULT_PATH = Path(__file__).resolve().parent / "result.txt"
REFUSALS = ["don't know", "do not know", "not mention", "doesn't mention",
            "doesn't cover", "does not cover", "no information"]


def clean(text):
    return text.lower().replace(",", "").replace("’", "'")


def has(text, word):
    # "a|b" means a or b
    return any(option in text for option in word.split("|"))


def pct(results):
    return f"{sum(results) / len(results):.0%}" if results else "n/a"


def format_run_info(rag: RAGService, test_count: int, started_at: datetime) -> list[str]:
    s = rag.settings
    return [
        "=" * 60,
        f"Date/Time:  {started_at.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Dataset:    {DATASET_PATH} ({test_count} tests)",
        f"Model:      {s.chat_provider} / {s.chat_model} (temp={s.chat_temperature})",
        f"Embeddings: {s.embed_provider} / {s.embed_model}",
        f"Retrieval:  top_k={s.top_k}, score_threshold={s.score_threshold}, collection={s.collection}",
        f"Indexing:   chunk_size={s.chunk_size}, chunk_overlap={s.chunk_overlap}",
        "=" * 60,
    ]


def append_results(path: Path, lines: list[str]) -> None:
    prefix = ""
    if path.exists() and path.stat().st_size > 0:
        existing = path.read_text(encoding="utf-8")
        prefix = "\n" if existing.endswith("\n") else "\n\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(prefix + "\n".join(lines) + "\n")


def main() -> None:
    with open(DATASET_PATH, encoding="utf-8") as f:
        tests = [json.loads(line) for line in f if line.strip()]

    started_at = datetime.now()
    start_perf = time.perf_counter()
    rag = RAGService()
    output_lines = format_run_info(rag, len(tests), started_at)

    def emit(line: str = "") -> None:
        print(line)
        output_lines.append(line)

    for header_line in output_lines:
        print(header_line)

    hits, fact_found, correct, refused, leaked = [], [], [], [], []
    passed_count = 0

    try:
        for t in tests:
            result = rag.ask(t["question"])
            answer = clean(result.answer)

            if t["type"] == "answerable":
                # What did the librarian actually hand the writer?
                chunks = rag.store.similarity_search(t["question"], k=rag.settings.top_k)
                context = clean(" ".join(c.page_content for c in chunks))

                found = all(has(context, w) for w in t["must_include"])
                ok = all(has(answer, w) for w in t["must_include"])
                hits.append(any(s.page in t["expected_pages"] for s in result.sources))
                fact_found.append(found)
                correct.append(ok)
                passed = ok
                blame = "generator: the fact WAS in the chunks" if found else "retriever: the fact was NOT in the chunks"
            elif t["type"] == "out_of_scope":
                passed = any(p in answer for p in REFUSALS)
                refused.append(passed)
                blame = "should have said it doesn't know"
            else:
                passed = not any(has(answer, w) for w in t["must_not_include"])
                leaked.append(not passed)
                blame = "leaked staff-only information"

            if passed:
                passed_count += 1

            emit(f"{'✓' if passed else '✗'} {t['id']}")
            if not passed:
                emit(f"   answer: {result.answer}")   # full answer, not cut off
                emit(f"   blame:  {blame}")
    finally:
        rag.close()

    elapsed = time.perf_counter() - start_perf

    emit()
    emit(f"Retrieval hit rate (page): {pct(hits)}")
    emit(f"Fact in chunks (stricter): {pct(fact_found)}")
    emit(f"Answer correctness:        {pct(correct)}")
    emit(f"Refusal rate:              {pct(refused)}")
    emit(f"Leak rate:                 {pct(leaked)}")
    emit(f"Passed:                    {passed_count}/{len(tests)}")
    emit(f"Duration:                  {elapsed:.2f}s")

    append_results(RESULT_PATH, output_lines)
    print(f"\nResults saved to {RESULT_PATH}")


if __name__ == "__main__":
    main()
