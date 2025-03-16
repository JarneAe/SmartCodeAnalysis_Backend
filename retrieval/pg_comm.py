import psycopg
from dotenv import load_dotenv
import os

load_dotenv()

db_params = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT"), 5432)
}

def get_codebase(codebase_id: str) -> {}:
    """
    Gets an entire codebase from the database.

    Args:
        codebase_id: The GUID of the codebase

    Returns:
        dict: Codebase with nested dictionaries for folders.

    """
    root_folder = get_root_folder_of_codebase(codebase_id)
    return get_children_of_folder(root_folder)

def get_root_folder_of_codebase(codebase_id: str) -> str:
    """
    Gets the GUID of the root folder associated with the provided codebase.

    Args:
        codebase_id: The GUID of the codebase.

    Returns:
        str: GUID of the root folder.
    """

    conn: psycopg.Connection
    with psycopg.connect(**db_params) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT "RootFolderId" FROM "Codebases" WHERE "Codebases"."Id" = %s
                """, (codebase_id,))

            try:
                return cur.fetchone()[0]
            except TypeError:
                raise ValueError(f"codebase {codebase_id} not found")


def get_children_of_folder(folder_id: str) -> {}:
    """
    Gets all children of a folder. If there are folders within the provided folder, its children will
    recursively be included as well.

    Args:
        folder_id: The GUID of the folder

    Returns:
        dict: The recursively retrieved children of the folder.
    """
    folder = {}

    conn: psycopg.Connection
    with psycopg.connect(**db_params) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                    SELECT * FROM "Files" 
                    WHERE ("Files"."Discriminator" LIKE 'CodeFile' OR "Files"."Discriminator" LIKE 'Folder')
                    AND "Files"."FolderId" = %s
                    """, (folder_id,))

            for row in cur.fetchall():
                if row[2] == "CodeFile":
                    folder[row[1]] = row[4]
                elif row[2] == "Folder":
                    folder[row[1]] = {"id": row[0]}

    for item in folder:
        if isinstance(folder[item], dict):
            folder[item] = get_children_of_folder(folder[item]["id"])

    return folder
