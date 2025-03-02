from retrieval.util import get_chunked_codebase

cb = get_chunked_codebase("8ab6b048-c99d-4c78-8955-1ef98fdb4f74")
print(cb)

content = [chunk['text'] for file in cb for chunk in file]

print(content)
