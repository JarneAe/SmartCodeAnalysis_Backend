from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

def chunk_file(file_name: str, content: bytes) -> list[dict[str, int | str]]:
    """
    Converts a code file to chunks using a language aware text splitter.

    Args:
        file_name: The name of the file
        content: The content of the file

    Returns:
        list[dict[str, int | str]]: the code file divided in chunks, annotated with metadata.
    """

    text = content.decode("utf-8")

    splitter = RecursiveCharacterTextSplitter.from_language(
        language=file_name_to_splitter_language(file_name),
        chunk_size=200,
        chunk_overlap=30
    )

    chunks = splitter.split_text(text)

    return [
        {"text": chunk, "file_name": file_name, "chunk_index": i}
        for i, chunk in enumerate(chunks)
    ]



def flatten_codebase(codebase: dict, prefix="") -> dict:
    """
    Flattens the codebase received from the database into a flat dictionary with directories in the name.

    Args:
        codebase: a dictionary with nested dictionaries representing directories
        prefix: the prefix to add to the files, usually the directory you're currently in.

    Returns:
         dict: A flattened representation of the codebase.
    """

    new_prefix = prefix + "/" if prefix != "" else ""
    flattened_codebase = dict()
    for key in codebase:
        if isinstance(codebase[key], dict):
            children = flatten_codebase(codebase[key], new_prefix + key)
            flattened_codebase = flattened_codebase | children # Merges both dicts
        else:
            flattened_codebase[new_prefix+key] = codebase[key]

    return flattened_codebase


def file_name_to_splitter_language(file_name: str) -> Language:
    """
    Converts a filename (e.g. main.py) to the equivalent langchain Language enum object using the file extension.

    Args:
        file_name: a filename (e.g. main.py, Program.cs...)

    Returns:
         Language: a Language enum object that matches that filename.
    """

    # TODO: expand this match case
    extension = file_name.split(".")[-1]
    match extension:
        case ".cs":
            return Language.CSHARP
        case ".java":
            return Language.JAVA
        case ".py":
            return Language.PYTHON
        case ".js":
            return Language.JS
        case ".ts":
            return  Language.TS

        # I don't really know what else to do with the default case.
        case _:
            return Language.JAVA
