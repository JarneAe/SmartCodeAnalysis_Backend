from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

def chunk_file(file_name: str, content: str) -> [str]:
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=file_name_to_splitter_language(file_name),
        chunk_size=200,
        chunk_overlap=30
    )

    return splitter.split_text(content)

# TODO: expand this match case
def file_name_to_splitter_language(name: str) -> Language:
    extension = name.split(".")[-1]
    match extension:
        case ".cs":
            return Language.CSHARP
        case ".java":
            return Language.JAVA
        case ".py":
            return Language.PYTHON
        case ".js":
            return Language.JS