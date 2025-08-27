def format_docs_with_citations(docs):
    parts = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        pg = d.metadata.get("page", "?")
        parts.append(f"[{i}] {d.page_content}\n(출처: {src}, 페이지 {pg})")
    return "\n\n".join(parts)